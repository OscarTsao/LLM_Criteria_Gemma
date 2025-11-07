"""Threshold search for multi-label evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class ThresholdResult:
    global_threshold: float
    macro_f1_global: float
    per_class_thresholds: Dict[str, float]
    per_class_f1: Dict[str, float]
    macro_f1_per_class: float


def search_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    grid_size: int = 1001,
) -> ThresholdResult:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_global = 0.5
    best_macro_f1 = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_global = float(threshold)

    per_class_thresholds: Dict[str, float] = {}
    per_class_scores: Dict[str, float] = {}
    for idx, label in enumerate(label_names):
        column = probabilities[:, idx]
        target = labels[:, idx]
        best_thr = 0.5
        best_score = -1.0
        for threshold in thresholds:
            preds = (column >= threshold).astype(int)
            score = f1_score(target, preds, zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_thr = float(threshold)
        per_class_thresholds[label] = best_thr
        per_class_scores[label] = best_score

    threshold_array = np.array([per_class_thresholds[label] for label in label_names])
    per_class_preds = (probabilities >= threshold_array).astype(int)
    macro_f1_per_class = f1_score(labels, per_class_preds, average="macro", zero_division=0)

    return ThresholdResult(
        global_threshold=best_global,
        macro_f1_global=float(best_macro_f1),
        per_class_thresholds=per_class_thresholds,
        per_class_f1=per_class_scores,
        macro_f1_per_class=float(macro_f1_per_class),
    )
