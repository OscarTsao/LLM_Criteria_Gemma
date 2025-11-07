"""ReDSM5 sentence-level dataset utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

LABEL_NAMES: tuple[str, ...] = (
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
    "SPECIAL_CASE",
)
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABEL_NAMES)}
NUM_LABELS: int = len(LABEL_NAMES)


@dataclass(frozen=True)
class SentenceRecord:
    post_id: str
    sentence_id: str
    position: int
    text: str
    labels: np.ndarray
    post_labels: np.ndarray


@dataclass(frozen=True)
class PostRecord:
    post_id: str
    sentences: List[SentenceRecord]
    labels: np.ndarray
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Example:
    text: str
    labels: np.ndarray
    post_id: str
    sentence_id: Optional[str]
    position: Optional[int]
    post_labels: np.ndarray


class RedSM5Dataset(Dataset):
    """Torch dataset that emits flattened sentence or post examples."""

    def __init__(self, examples: Sequence[Example], level: str = "sentence") -> None:
        if level not in {"sentence", "post"}:
            raise ValueError(f"Unsupported level '{level}'.")
        self.level = level
        self._examples = list(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        return {
            "text": example.text,
            "labels": torch.from_numpy(example.labels.astype(np.float32)),
            "post_labels": torch.from_numpy(example.post_labels.astype(np.float32)),
            "meta": {
                "post_id": example.post_id,
                "sentence_id": example.sentence_id,
                "position": example.position,
                "level": self.level,
            },
        }

    @staticmethod
    def from_posts(posts: Sequence[PostRecord], level: str = "sentence") -> "RedSM5Dataset":
        examples: List[Example] = []
        if level == "sentence":
            for post in posts:
                for sentence in post.sentences:
                    examples.append(
                        Example(
                            text=sentence.text,
                            labels=sentence.labels.copy(),
                            post_id=post.post_id,
                            sentence_id=sentence.sentence_id,
                            position=sentence.position,
                            post_labels=post.labels.copy(),
                        )
                    )
        else:
            for post in posts:
                text = "\n".join(sentence.text for sentence in post.sentences)
                examples.append(
                    Example(
                        text=text,
                        labels=post.labels.copy(),
                        post_id=post.post_id,
                        sentence_id=None,
                        position=None,
                        post_labels=post.labels.copy(),
                    )
                )
        return RedSM5Dataset(examples=examples, level=level)


class RedSM5DataCollator:
    """Tokenises and pads a batch of examples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        padding: str | bool = True,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        texts = [feature["text"] for feature in features]
        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = torch.stack([feature["labels"] for feature in features]).float()
        post_labels = torch.stack([feature["post_labels"] for feature in features]).float()
        meta = [feature["meta"] for feature in features]
        return {**tokenized, "labels": labels, "post_labels": post_labels, "meta": meta}


def _load_annotation_table(data_dir: Path) -> pd.DataFrame:
    candidates = [
        "redsm5_annotations.parquet",
        "redsm5_annotations.feather",
        "redsm5_annotations.csv",
        "annotations.parquet",
        "annotations.csv",
        "redsm5_annotations.json",
        "redsm5_annotations.jsonl",
    ]
    for name in candidates:
        path = data_dir / name
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".feather":
            return pd.read_feather(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return pd.json_normalize(data)
        if path.suffix == ".jsonl":
            records: List[MutableMapping[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return pd.DataFrame(records)
    raise FileNotFoundError(f"Could not locate annotations in {data_dir}.")


def _normalise_status(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return 0
    value_str = str(value).strip().lower()
    mapping = {
        "1": 1,
        "0": 0,
        "positive": 1,
        "present": 1,
        "yes": 1,
        "true": 1,
        "negative": 0,
        "absent": 0,
        "no": 0,
        "false": 0,
    }
    if value_str in mapping:
        return mapping[value_str]
    try:
        return int(float(value_str))
    except ValueError:
        LOGGER.warning("Unknown status value '%s'; defaulting to 0.", value)
        return 0


def _normalise_symptom(symptom: Any) -> str:
    if symptom is None:
        return "SPECIAL_CASE"
    token = str(symptom).strip().upper().replace(" ", "_").replace("-", "_")
    if token not in LABEL_TO_ID:
        LOGGER.warning("Unknown symptom '%s'; mapping to SPECIAL_CASE.", symptom)
        return "SPECIAL_CASE"
    return token


def _ensure_sentence_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "sentence_id" in df.columns and df["sentence_id"].notnull().all():
        return df.copy()
    df = df.copy()
    df["sentence_id"] = df.groupby("post_id").cumcount().astype(str)
    return df


def _compute_post_labels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    post_labels: Dict[str, np.ndarray] = {}
    for post_id, group in df.groupby("post_id"):
        vec = np.zeros(NUM_LABELS, dtype=np.float32)
        for symptom, status in zip(group["DSM5_symptom"], group["status"]):
            idx = LABEL_TO_ID.get(symptom)
            if idx is not None and status == 1:
                vec[idx] = 1.0
        post_labels[str(post_id)] = vec
    return post_labels


def _build_post_records(df: pd.DataFrame) -> List[PostRecord]:
    df = df.dropna(subset=["sentence_text"]).copy()
    if "post_id" not in df.columns:
        raise KeyError("Expected column 'post_id' in annotations.")
    df["post_id"] = df["post_id"].astype(str)
    df["status"] = df["status"].map(_normalise_status)
    df["DSM5_symptom"] = df["DSM5_symptom"].map(_normalise_symptom)
    df = _ensure_sentence_ids(df)
    df = df.sort_values(["post_id", "sentence_id"])

    post_labels = _compute_post_labels(df)
    posts: List[PostRecord] = []

    for post_id, group in df.groupby("post_id"):
        sentences: List[SentenceRecord] = []
        for position, (sentence_id, sentence_df) in enumerate(group.groupby("sentence_id"), start=0):
            text_series = sentence_df["sentence_text"].dropna()
            if text_series.empty:
                continue
            text = str(text_series.iloc[0])
            labels = np.zeros(NUM_LABELS, dtype=np.float32)
            for symptom, status in zip(sentence_df["DSM5_symptom"], sentence_df["status"]):
                idx = LABEL_TO_ID.get(symptom)
                if idx is not None and status == 1:
                    labels[idx] = 1.0
            sentences.append(
                SentenceRecord(
                    post_id=str(post_id),
                    sentence_id=str(sentence_id),
                    position=position,
                    text=text,
                    labels=labels,
                    post_labels=post_labels[str(post_id)].copy(),
                )
            )
        if not sentences:
            continue
        posts.append(
            PostRecord(
                post_id=str(post_id),
                sentences=sentences,
                labels=post_labels[str(post_id)].copy(),
                metadata={"sentence_count": len(sentences)},
            )
        )
    return posts


def load_redsm5_posts(data_dir: Path | str) -> List[PostRecord]:
    data_path = Path(data_dir)
    df = _load_annotation_table(data_path)
    posts = _build_post_records(df)
    if not posts:
        raise ValueError("No posts parsed from annotations.")
    return posts


def filter_posts(posts: Sequence[PostRecord], allowed_post_ids: Sequence[str]) -> List[PostRecord]:
    allowed = {str(pid) for pid in allowed_post_ids}
    return [post for post in posts if post.post_id in allowed]


def compute_class_distribution(posts: Sequence[PostRecord], level: str = "sentence") -> np.ndarray:
    counts = np.zeros(NUM_LABELS, dtype=np.float64)
    if level == "sentence":
        for post in posts:
            for sentence in post.sentences:
                counts += sentence.labels
    else:
        for post in posts:
            counts += post.labels
    return counts


def create_inverse_frequency_weights(distribution: np.ndarray) -> torch.Tensor:
    freq = distribution.copy()
    freq[freq == 0] = 1.0
    inv = 1.0 / freq
    inv = inv / inv.sum() * len(freq)
    return torch.tensor(inv, dtype=torch.float32)
