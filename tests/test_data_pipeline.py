from pathlib import Path

import numpy as np
import torch

from src.data.redsm5_dataset import (
    LABEL_NAMES,
    PostRecord,
    RedSM5DataCollator,
    RedSM5Dataset,
    SentenceRecord,
)


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt", **kwargs):
        input_ids = []
        attention_masks = []
        for text in texts:
            tokens = text.split()
            ids = [min(len(tokens), max_length)] * min(len(tokens), max_length)
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_masks.append(mask)
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]
        padded_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_masks]
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }


def _make_post(post_id: str, sentences: list[tuple[str, str, list[int]]]) -> PostRecord:
    sentence_records: list[SentenceRecord] = []
    labels = np.zeros(len(LABEL_NAMES), dtype=np.float32)
    for position, (sentence_id, text, positive_indices) in enumerate(sentences):
        sent_labels = np.zeros(len(LABEL_NAMES), dtype=np.float32)
        for idx in positive_indices:
            sent_labels[idx] = 1.0
            labels[idx] = 1.0
        sentence_records.append(
            SentenceRecord(
                post_id=post_id,
                sentence_id=sentence_id,
                position=position,
                text=text,
                labels=sent_labels,
                post_labels=labels.copy(),
            )
        )
    return PostRecord(post_id=post_id, sentences=sentence_records, labels=labels.copy(), metadata={"sentence_count": len(sentence_records)})


def test_sentence_dataset_and_collator_no_leak():
    posts = [
        _make_post(
            "p1",
            [
                ("s1", "Feeling down", [0]),
                ("s2", "Lost interest", [1]),
            ],
        ),
        _make_post(
            "p2",
            [
                ("s1", "Sleeping well", []),
                ("s2", "Appetite reduced", [2]),
            ],
        ),
    ]

    train_dataset = RedSM5Dataset.from_posts(posts[:1], level="sentence")
    val_dataset = RedSM5Dataset.from_posts(posts[1:], level="sentence")

    train_post_ids = {train_dataset[i]["meta"]["post_id"] for i in range(len(train_dataset))}
    val_post_ids = {val_dataset[i]["meta"]["post_id"] for i in range(len(val_dataset))}
    assert train_post_ids == {"p1"}
    assert val_post_ids == {"p2"}

    collator = RedSM5DataCollator(DummyTokenizer(), max_length=32)
    batch = collator([train_dataset[0], train_dataset[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].dtype == torch.float32
    assert all("post_id" in meta for meta in batch["meta"])
