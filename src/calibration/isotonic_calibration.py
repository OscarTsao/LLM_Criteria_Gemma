"""
Isotonic Regression calibration for multi-class classification.

Reference: Zadrozny & Elkan "Transforming classifier scores into accurate
multiclass probability estimates" (KDD 2002)
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class IsotonicCalibration:
    """
    Isotonic regression calibration for multi-class classification.

    Fits a separate isotonic regression model for each class using
    one-vs-rest approach.

    Args:
        num_classes: Number of classes
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.calibrators = [IsotonicRegression(out_of_bounds='clip')
                           for _ in range(num_classes)]
        self.fitted = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ) -> None:
        """
        Fit isotonic regression calibrators for each class.

        Args:
            probs: (N, num_classes) probability predictions
            labels: (N,) integer labels
            verbose: Print progress
        """
        # Validate inputs
        assert probs.shape[0] == labels.shape[0], "Shape mismatch"
        assert probs.shape[1] == self.num_classes, f"Expected {self.num_classes} classes"
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"

        # Create one-hot encoded labels
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1

        # Fit isotonic regression for each class
        for class_idx in range(self.num_classes):
            class_probs = probs[:, class_idx]
            class_labels = one_hot[:, class_idx]

            # Handle edge case: constant probabilities
            if np.unique(class_probs).size == 1:
                logger.warning(f"Class {class_idx}: constant probabilities, "
                             f"calibrator may not be useful")

            self.calibrators[class_idx].fit(class_probs, class_labels)

        self.fitted = True

        if verbose:
            logger.info(f"Isotonic calibration fitted for {self.num_classes} classes")

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply learned isotonic calibration to probabilities.

        Args:
            probs: (N, num_classes) probability predictions

        Returns:
            Calibrated probabilities (N, num_classes), re-normalized
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        assert probs.shape[1] == self.num_classes, f"Expected {self.num_classes} classes"

        # Apply isotonic regression per class
        calibrated = np.zeros_like(probs)
        for class_idx in range(self.num_classes):
            calibrated[:, class_idx] = self.calibrators[class_idx].predict(probs[:, class_idx])

        # Clip to [0, 1] range
        calibrated = np.clip(calibrated, 0, 1)

        # Re-normalize to sum to 1 (isotonic doesn't preserve normalization)
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    def test_constant_edge_case(self, const_prob: float = 0.5) -> float:
        """
        Test isotonic regression on constant probability input.

        Args:
            const_prob: Constant probability value

        Returns:
            Calibrated probability (should be close to input)
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted")

        test_probs = np.full((1, self.num_classes), const_prob / self.num_classes)
        calibrated = self.calibrate(test_probs)
        return calibrated[0]
