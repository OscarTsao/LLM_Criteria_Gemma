"""Metric utilities for ReDSM5 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)


@dataclass
class MetricResult:
    macro_auprc: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    per_class_pr: Dict[str, Dict[str, List[float]]]


def compute_macro_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    scores: List[float] = []
    for label_idx in range(y_true.shape[1]):
        positives = y_true[:, label_idx].sum()
        if positives == 0:
            continue
        try:
            score = average_precision_score(y_true[:, label_idx], y_score[:, label_idx])
        except ValueError:
            score = 0.0
        scores.append(float(score))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_macro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    if label_names is None:
        label_names = [str(idx) for idx in range(y_true.shape[1])]
    per_class_map = {
        str(label_names[idx]): float(per_class[idx]) for idx in range(len(label_names))
    }
    return {"macro": macro, "per_class": per_class_map}


def compute_per_class_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
    max_points: int = 200,
) -> Dict[str, Dict[str, List[float]]]:
    curves: Dict[str, Dict[str, List[float]]] = {}
    for idx, label in enumerate(label_names):
        precision, recall, thresholds = precision_recall_curve(y_true[:, idx], y_score[:, idx])
        if max_points is not None and len(precision) > max_points:
            indices = np.linspace(0, len(precision) - 1, max_points).astype(int)
            precision = precision[indices]
            recall = recall[indices]
            thresholds = thresholds[np.clip(indices[:-1], 0, len(thresholds) - 1)]
        curves[label] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist() if thresholds.size > 0 else [],
        }
    return curves


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    label_names: Sequence[str],
    pr_points: int = 200,
) -> MetricResult:
    macro_auprc = compute_macro_auprc(y_true, y_score)
    f1_result = compute_macro_f1(y_true, y_score, threshold=threshold, label_names=label_names)
    pr_curves = compute_per_class_pr(y_true, y_score, label_names=label_names, max_points=pr_points)
    return MetricResult(
        macro_auprc=macro_auprc,
        macro_f1=f1_result["macro"],
        per_class_f1=f1_result["per_class"],
        per_class_pr=pr_curves,
    )
