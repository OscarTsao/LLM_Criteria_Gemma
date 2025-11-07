"""Post-level fold generation for ReDSM5."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .redsm5_dataset import LABEL_NAMES, PostRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_posts: List[str]
    val_posts: List[str]


def _derive_stratify_keys(posts: Sequence[PostRecord]) -> np.ndarray:
    keys: List[str] = []
    for post in posts:
        positives = [LABEL_NAMES[idx] for idx, value in enumerate(post.labels) if value > 0.5]
        if not positives:
            keys.append("none")
        elif len(positives) == 1:
            keys.append(positives[0])
        else:
            keys.append("multi")
    return np.asarray(keys)


def _write_split(splits_dir: Path, split: FoldSplit, seed: int, stratified: bool) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold": split.fold,
        "seed": seed,
        "stratified": stratified,
        "train_posts": split.train_posts,
        "val_posts": split.val_posts,
    }
    with (splits_dir / f"fold_{split.fold}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def create_folds(
    posts: Sequence[PostRecord],
    splits_dir: Path,
    folds: int = 5,
    seed: int = 42,
) -> List[FoldSplit]:
    post_ids = np.asarray([post.post_id for post in posts])
    stratify_keys = _derive_stratify_keys(posts)
    counts = {key: int((stratify_keys == key).sum()) for key in np.unique(stratify_keys)}
    stratified = all(count >= folds for count in counts.values())
    if not stratified:
        LOGGER.warning("Insufficient samples for stratified %d-fold split. Falling back to unstratified KFold.", folds)
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(post_ids, stratify_keys)
        if stratified
        else KFold(n_splits=folds, shuffle=True, random_state=seed).split(post_ids)
    )
    splits: List[FoldSplit] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter):
        train_posts = post_ids[train_idx].tolist()
        val_posts = post_ids[val_idx].tolist()
        split = FoldSplit(fold=fold_idx, train_posts=train_posts, val_posts=val_posts)
        _write_split(splits_dir, split, seed, stratified)
        splits.append(split)
    return splits


def read_split(splits_dir: Path, fold: int) -> FoldSplit:
    path = splits_dir / f"fold_{fold}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FoldSplit(
        fold=int(payload["fold"]),
        train_posts=[str(pid) for pid in payload["train_posts"]],
        val_posts=[str(pid) for pid in payload["val_posts"]],
    )


def ensure_folds(
    posts: Sequence[PostRecord],
    splits_dir: Path,
    folds: int = 5,
    seed: int = 42,
) -> List[FoldSplit]:
    try:
        return [read_split(splits_dir, fold_idx) for fold_idx in range(folds)]
    except FileNotFoundError:
        LOGGER.info("Generating fold splits at %s", splits_dir)
        return create_folds(posts, splits_dir, folds=folds, seed=seed)


def fold_statistics(splits: Sequence[FoldSplit]) -> Dict[int, Dict[str, int]]:
    return {split.fold: {"train": len(split.train_posts), "val": len(split.val_posts)} for split in splits}
