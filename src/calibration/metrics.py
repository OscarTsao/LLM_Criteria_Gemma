"""
Calibration metrics and diagnostic plots.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) for multi-class classification.

    ECE measures the difference between confidence and accuracy.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        num_bins: Number of bins for calibration

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: (num_bins,) accuracy per bin
        bin_confidences: (num_bins,) average confidence per bin
        bin_counts: (num_bins,) number of samples per bin
    """
    # Get predicted class and confidence
    pred_labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (pred_labels == labels).astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    ece = 0.0
    total_samples = len(confidences)

    for bin_idx in range(num_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_lowers[bin_idx]) & (confidences < bin_uppers[bin_idx])
        bin_count = in_bin.sum()

        if bin_count > 0:
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()

            bin_accuracies[bin_idx] = bin_accuracy
            bin_confidences[bin_idx] = bin_confidence
            bin_counts[bin_idx] = bin_count

            # Weighted contribution to ECE
            ece += (bin_count / total_samples) * abs(bin_accuracy - bin_confidence)

    return ece, bin_accuracies, bin_confidences, bin_counts


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create reliability diagram data for visualization.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        num_bins: Number of bins
        save_path: Optional path to save plot

    Returns:
        bin_confidences: (num_bins,) x-axis values
        bin_accuracies: (num_bins,) y-axis values
        bin_counts: (num_bins,) bar heights
    """
    ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
        probs, labels, num_bins
    )

    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    if save_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

            # Plot bins
            ax.bar(
                bin_confidences,
                bin_accuracies,
                width=1.0/num_bins,
                alpha=0.6,
                edgecolor='black',
                label='Model'
            )

            ax.set_xlabel('Confidence', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            logger.info(f"Reliability diagram saved to {save_path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")

    return bin_confidences, bin_accuracies, bin_counts


def classwise_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15
) -> np.ndarray:
    """
    Compute per-class ECE for multi-class classification.

    Args:
        probs: (N, num_classes) probability predictions
        labels: (N,) integer labels
        num_bins: Number of bins

    Returns:
        ece_per_class: (num_classes,) ECE for each class
    """
    num_classes = probs.shape[1]
    ece_per_class = np.zeros(num_classes)

    # Convert to one-hot
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1

    for class_idx in range(num_classes):
        class_probs = probs[:, class_idx]
        class_labels = one_hot[:, class_idx]

        # Bin the probabilities
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(class_probs)

        for bin_idx in range(num_bins):
            in_bin = (class_probs >= bin_lowers[bin_idx]) & (class_probs < bin_uppers[bin_idx])
            bin_count = in_bin.sum()

            if bin_count > 0:
                bin_accuracy = class_labels[in_bin].mean()
                bin_confidence = class_probs[in_bin].mean()
                ece += (bin_count / total_samples) * abs(bin_accuracy - bin_confidence)

        ece_per_class[class_idx] = ece

    return ece_per_class
