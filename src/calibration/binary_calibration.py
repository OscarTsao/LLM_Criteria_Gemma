"""
Binary calibration utilities for post-criterion matching.

Simpler than multi-class calibration since we only have 2 classes.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def apply_temperature_scaling_binary(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Apply temperature scaling to binary logits.

    Args:
        logits: (N, 2) array of logits
        labels: (N,) array of binary labels (0 or 1)
        temperature: Fixed temperature (if None, will be optimized)

    Returns:
        calibrated_probs: (N,) array of calibrated probabilities for class 1
        temperature: Optimal or provided temperature
    """
    import torch
    import torch.nn as nn

    if temperature is None:
        # Optimize temperature on validation set
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        # Initialize temperature
        temp = torch.nn.Parameter(torch.ones(1))

        # Optimizer
        optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)

        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits_t / torch.clamp(temp, min=0.01, max=100.0)
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        temperature = temp.item()
        logger.info(f"Optimal temperature: {temperature:.4f}")

    # Apply temperature
    logits_t = torch.from_numpy(logits).float()
    calibrated_logits = logits_t / temperature
    calibrated_probs = torch.softmax(calibrated_logits, dim=1)[:, 1].numpy()

    return calibrated_probs, temperature


def apply_platt_scaling(
    scores: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Apply Platt scaling (logistic regression calibration) for binary classification.

    Args:
        scores: (N,) uncalibrated scores (can be logits or probabilities)
        labels: (N,) binary labels

    Returns:
        calibrated_probs: (N,) calibrated probabilities
    """
    # Fit logistic regression
    platt = LogisticRegression()
    platt.fit(scores.reshape(-1, 1), labels)

    # Calibrate
    calibrated_probs = platt.predict_proba(scores.reshape(-1, 1))[:, 1]

    return calibrated_probs


def apply_isotonic_regression_binary(
    scores: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Apply isotonic regression for binary classification.

    Args:
        scores: (N,) uncalibrated scores
        labels: (N,) binary labels

    Returns:
        calibrated_probs: (N,) calibrated probabilities
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(scores, labels)

    calibrated_probs = iso.predict(scores)

    return calibrated_probs


def compute_binary_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error for binary classification.

    Args:
        probs: (N,) predicted probabilities for positive class
        labels: (N,) binary labels

    Returns:
        ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)

    return ece


def plot_binary_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram for binary classification.

    Args:
        probs: (N,) predicted probabilities
        labels: (N,) binary labels
        num_bins: Number of bins
        save_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=num_bins, strategy='uniform'
        )

        # Compute ECE
        ece = compute_binary_ece(probs, labels, num_bins)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        # Actual calibration
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-',
               label=f'Model (ECE={ece:.4f})', linewidth=2, markersize=8)

        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Binary Reliability Diagram', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Reliability diagram saved to {save_path}")

        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plot")


def optimize_binary_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Optimize decision threshold for binary classification.

    Args:
        probs: (N,) predicted probabilities
        labels: (N,) binary labels
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')

    Returns:
        best_threshold: Optimal threshold
        best_score: Score at optimal threshold
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0, 1, 100)
    best_score = -1
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'balanced_accuracy':
            # (TPR + TNR) / 2
            tp = ((preds == 1) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()

            tpr = tp / (tp + fn + 1e-8)
            tnr = tn / (tn + fp + 1e-8)
            score = (tpr + tnr) / 2
        elif metric == 'youden':
            # Youden's J statistic = TPR + TNR - 1
            tp = ((preds == 1) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()

            tpr = tp / (tp + fn + 1e-8)
            tnr = tn / (tn + fp + 1e-8)
            score = tpr + tnr - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
