"""Sentence to post-level aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


@dataclass
class AggregationResult:
    """Aggregated probabilities grouped by post."""

    post_ids: List[str]
    probabilities: np.ndarray
    weights: Dict[str, np.ndarray]
    sentence_indices: Dict[str, List[int]]


def group_indices_by_post(meta: Sequence[Mapping[str, object]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, entry in enumerate(meta):
        post_id = str(entry.get("post_id", ""))
        groups.setdefault(post_id, []).append(idx)
    return groups


def aggregate_probabilities(
    probabilities: np.ndarray,
    meta: Sequence[Mapping[str, object]],
    strategy: str = "max",
    temperature: float = 1.0,
) -> AggregationResult:
    groups = group_indices_by_post(meta)
    post_ids: List[str] = []
    post_probs: List[np.ndarray] = []
    weights: Dict[str, np.ndarray] = {}
    for post_id, indices in groups.items():
        sentence_probs = probabilities[indices]
        if strategy == "max":
            agg = sentence_probs.max(axis=0)
            weight_vector = np.zeros(len(indices), dtype=np.float32)
            focus_idx = int(np.argmax(sentence_probs.mean(axis=1)))
            weight_vector[focus_idx] = 1.0
        elif strategy == "mean":
            agg = sentence_probs.mean(axis=0)
            weight_vector = np.ones(len(indices), dtype=np.float32) / len(indices)
        elif strategy == "attention":
            scores = sentence_probs.mean(axis=1)
            temp = max(temperature, 1e-3)
            scaled = scores / temp
            scaled -= scaled.max()
            exp_scores = np.exp(scaled)
            denom = exp_scores.sum()
            if denom == 0.0:
                weight_vector = np.ones(len(indices), dtype=np.float32) / len(indices)
            else:
                weight_vector = exp_scores / denom
            agg = (weight_vector[:, None] * sentence_probs).sum(axis=0)
        else:
            raise ValueError(f"Unsupported aggregation strategy '{strategy}'.")
        post_ids.append(post_id)
        post_probs.append(agg)
        weights[post_id] = weight_vector
    stacked = np.vstack(post_probs) if post_probs else np.zeros((0, probabilities.shape[1]), dtype=np.float32)
    return AggregationResult(post_ids=post_ids, probabilities=stacked, weights=weights, sentence_indices=groups)


def aggregate_labels(
    labels: np.ndarray,
    meta: Sequence[Mapping[str, object]],
) -> np.ndarray:
    groups = group_indices_by_post(meta)
    post_labels: List[np.ndarray] = []
    for post_id, indices in groups.items():
        sentence_labels = labels[indices]
        post_label = sentence_labels.max(axis=0)
        post_labels.append(post_label)
    if not post_labels:
        return np.zeros((0, labels.shape[1]), dtype=labels.dtype)
    return np.vstack(post_labels)
