"""Metric reporting and summary utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from src.training.metrics import (
    compute_macro_auprc,
    compute_per_class_pr,
)


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    confidence: float
    accuracy: float
    count: int


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> tuple[float, List[ReliabilityBin]]:
    probs = probabilities.flatten()
    truths = labels.flatten()
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probs)
    ece = 0.0
    hist: List[ReliabilityBin] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(truths[mask].mean())
        ece += (count / total) * abs(acc - conf)
        hist.append(ReliabilityBin(lower=lower, upper=upper, confidence=conf, accuracy=acc, count=count))
    return float(ece), hist


def coverage_risk_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Sequence[float],
) -> List[Dict[str, float]]:
    max_scores = probabilities.max(axis=1)
    curve: List[Dict[str, float]] = []
    for threshold in thresholds:
        mask = max_scores >= threshold
        coverage = float(mask.mean())
        if coverage == 0.0:
            risk = 0.0
        else:
            preds = (probabilities[mask] >= 0.5).astype(int)
            risk = 1.0 - float(f1_score(labels[mask], preds, average="macro", zero_division=0))
        curve.append({"threshold": float(threshold), "coverage": coverage, "risk": risk})
    return curve


def per_class_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    global_threshold: float,
    per_class_thresholds: Dict[str, float],
    per_class_f1: Dict[str, float],
) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for idx, label in enumerate(label_names):
        column_probs = probabilities[:, idx]
        column_labels = labels[:, idx]
        positives = column_labels.sum()
        auprc = average_precision_score(column_labels, column_probs) if positives else 0.0
        f1_global = f1_score(
            column_labels,
            (column_probs >= global_threshold).astype(int),
            zero_division=0,
        )
        summary.append(
            {
                "label": label,
                "auprc": float(auprc),
                "f1_global": float(f1_global),
                "threshold_global": float(global_threshold),
                "f1_opt": float(per_class_f1[label]),
                "threshold_opt": float(per_class_thresholds[label]),
            }
        )
    return summary


def generate_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    global_threshold: float,
    per_class_thresholds: Dict[str, float],
    per_class_f1: Dict[str, float],
    macro_f1_per_class: float,
    ece_bins: int = 15,
) -> Dict[str, object]:
    macro_auprc = compute_macro_auprc(labels, probabilities)
    global_preds = (probabilities >= global_threshold).astype(int)
    macro_f1_global = float(f1_score(labels, global_preds, average="macro", zero_division=0))
    ece_value, reliability = expected_calibration_error(probabilities, labels, bins=ece_bins)
    pr_curves = compute_per_class_pr(labels, probabilities, label_names)
    coverage_thresholds = np.linspace(0.0, 1.0, 21)
    coverage_risk = coverage_risk_curve(probabilities, labels, coverage_thresholds)
    summary = per_class_summary(
        probabilities,
        labels,
        label_names,
        global_threshold=global_threshold,
        per_class_thresholds=per_class_thresholds,
        per_class_f1=per_class_f1,
    )
    return {
        "macro_auprc": float(macro_auprc),
        "macro_f1_global": macro_f1_global,
        "macro_f1_per_class": float(macro_f1_per_class),
        "ece": ece_value,
        "reliability": [r.__dict__ for r in reliability],
        "coverage_risk": coverage_risk,
        "per_class": summary,
        "per_class_pr": pr_curves,
    }
