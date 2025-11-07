"""
Threshold optimization for multi-class classification.

Implements per-class threshold search to maximize F1 score.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def optimize_thresholds_per_class(
    probs: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 100,
    metric: str = 'f1'
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Optimize decision threshold for each class independently.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels (multi-class)
        num_thresholds: Number of threshold values to try
        metric: Metric to optimize ('f1' or 'balanced_accuracy')

    Returns:
        best_thresholds: (num_classes,) optimal threshold per class
        metrics_at_best: Dict of metrics at best thresholds
    """
    num_classes = probs.shape[1]
    best_thresholds = np.full(num_classes, 0.5)  # Default
    metrics_at_best = {}

    # Convert to one-vs-rest binary labels
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1

    threshold_grid = np.linspace(0.0, 1.0, num_thresholds)

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = one_hot[:, class_idx]

        # Handle edge case: no positive examples
        if class_labels.sum() == 0:
            logger.warning(f"Class {class_idx}: No positive examples, using default threshold 0.5")
            best_thresholds[class_idx] = 0.5
            continue

        best_score = -1.0
        best_threshold = 0.5

        for threshold in threshold_grid:
            predictions = (class_probs >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(class_labels, predictions, zero_division=0)
            elif metric == 'balanced_accuracy':
                # Balanced accuracy = (TPR + TNR) / 2
                tp = ((predictions == 1) & (class_labels == 1)).sum()
                tn = ((predictions == 0) & (class_labels == 0)).sum()
                fp = ((predictions == 1) & (class_labels == 0)).sum()
                fn = ((predictions == 0) & (class_labels == 1)).sum()

                tpr = tp / (tp + fn + 1e-8)
                tnr = tn / (tn + fp + 1e-8)
                score = (tpr + tnr) / 2
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        best_thresholds[class_idx] = best_threshold
        metrics_at_best[class_idx] = best_score

    return best_thresholds, metrics_at_best


def compute_pr_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    save_arrays: bool = True
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute precision-recall curves for each class.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        save_arrays: Whether to save full precision/recall arrays

    Returns:
        Dict mapping class_idx to {precision, recall, thresholds, auprc}
    """
    num_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1

    pr_curves = {}

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = one_hot[:, class_idx]

        # Handle edge case: no positive examples
        if class_labels.sum() == 0:
            logger.warning(f"Class {class_idx}: No positive examples, skipping PR curve")
            pr_curves[class_idx] = {
                'precision': np.array([1.0]),
                'recall': np.array([0.0]),
                'thresholds': np.array([0.0]),
                'auprc': 0.0
            }
            continue

        precision, recall, thresholds = precision_recall_curve(class_labels, class_probs)
        auprc = auc(recall, precision)

        pr_curves[class_idx] = {
            'precision': precision if save_arrays else None,
            'recall': recall if save_arrays else None,
            'thresholds': thresholds if save_arrays else None,
            'auprc': auprc
        }

    return pr_curves


def compute_macro_auprc(pr_curves: Dict[int, Dict[str, np.ndarray]]) -> float:
    """
    Compute macro-averaged AUPRC across all classes.

    Args:
        pr_curves: Output from compute_pr_curves

    Returns:
        Macro-averaged AUPRC
    """
    auprcs = [curve['auprc'] for curve in pr_curves.values()]
    return np.mean(auprcs)


def compute_coverage_risk_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    confidence_thresholds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute coverage-risk curve for selective prediction.

    Allows model to abstain on low-confidence predictions.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        confidence_thresholds: Thresholds to test (default: linspace(0, 1, 100))

    Returns:
        coverages: (num_thresholds,) fraction of samples predicted
        risks: (num_thresholds,) error rate on predicted samples
        thresholds: (num_thresholds,) confidence thresholds used
    """
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(0, 1, 100)

    pred_labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (pred_labels == labels).astype(float)

    coverages = []
    risks = []

    for threshold in confidence_thresholds:
        # Only predict on samples above threshold
        mask = confidences >= threshold

        if mask.sum() == 0:
            coverage = 0.0
            risk = 1.0  # All predictions wrong (or undefined)
        else:
            coverage = mask.mean()
            risk = 1.0 - correct[mask].mean()  # Error rate

        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks), confidence_thresholds


def validate_threshold_stability(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bootstrap: int = 10,
    subsample_ratio: float = 0.8
) -> Dict[str, np.ndarray]:
    """
    Test stability of threshold optimization via bootstrap resampling.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        num_bootstrap: Number of bootstrap samples
        subsample_ratio: Fraction of data to use per bootstrap

    Returns:
        Dict with mean and std of thresholds across bootstrap samples
    """
    num_classes = probs.shape[1]
    n_samples = len(labels)
    subsample_size = int(n_samples * subsample_ratio)

    threshold_samples = np.zeros((num_bootstrap, num_classes))

    for boot_idx in range(num_bootstrap):
        # Random subsample with replacement
        indices = np.random.choice(n_samples, size=subsample_size, replace=True)
        boot_probs = probs[indices]
        boot_labels = labels[indices]

        # Optimize thresholds on bootstrap sample
        thresholds, _ = optimize_thresholds_per_class(boot_probs, boot_labels)
        threshold_samples[boot_idx] = thresholds

    return {
        'mean_thresholds': threshold_samples.mean(axis=0),
        'std_thresholds': threshold_samples.std(axis=0),
        'all_samples': threshold_samples
    }
