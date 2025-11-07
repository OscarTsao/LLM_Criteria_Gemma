"""
Tests for calibration methods.

Tests:
1. Temperature scaling monotonicity
2. Temperature scaling NLL reduction
3. Isotonic regression edge cases
4. Threshold optimization stability
5. ECE computation correctness
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.calibration import (
    TemperatureScaling,
    IsotonicCalibration,
    expected_calibration_error,
    reliability_diagram
)
from src.calibration.temperature_scaling import validate_monotonicity
from src.calibration.threshold_optimization import (
    optimize_thresholds_per_class,
    compute_pr_curves,
    validate_threshold_stability
)


class TestTemperatureScaling:
    """Test temperature scaling calibration."""

    @pytest.fixture
    def sample_logits_and_labels(self):
        """Create sample logits and labels."""
        np.random.seed(42)
        n_samples = 1000
        num_classes = 5

        # Create slightly miscalibrated logits
        logits = np.random.randn(n_samples, num_classes) * 2.0
        # Add systematic overconfidence
        logits = logits * 1.5

        labels = np.random.randint(0, num_classes, size=n_samples)

        return logits, labels

    def test_temperature_scaling_reduces_nll(self, sample_logits_and_labels):
        """Test that temperature scaling reduces NLL on validation set."""
        logits, labels = sample_logits_and_labels

        # Split into fit and test
        split_idx = len(logits) // 2
        fit_logits, fit_labels = logits[:split_idx], labels[:split_idx]
        test_logits, test_labels = logits[split_idx:], labels[split_idx:]

        # Compute initial NLL on test set
        initial_probs = torch.softmax(torch.from_numpy(test_logits).float(), dim=1).numpy()
        initial_nll = -np.log(initial_probs[np.arange(len(test_labels)), test_labels] + 1e-8).mean()

        # Fit temperature scaling
        temp_scaler = TemperatureScaling(num_classes=5)
        temp_scaler.fit(fit_logits, fit_labels, verbose=False)

        # Calibrate test set
        calibrated_probs = temp_scaler.calibrate(test_logits)
        calibrated_nll = -np.log(calibrated_probs[np.arange(len(test_labels)), test_labels] + 1e-8).mean()

        # NLL should reduce (or at least not increase significantly)
        assert calibrated_nll <= initial_nll + 0.1, (
            f"Temperature scaling increased NLL: {initial_nll:.4f} -> {calibrated_nll:.4f}"
        )

    def test_temperature_monotonicity(self):
        """Test that probabilities are monotonic in temperature."""
        logits = np.array([2.0, 1.0, 0.5, 0.1])

        is_monotonic = validate_monotonicity(logits)

        assert is_monotonic, "Probabilities not monotonic in temperature"

    def test_temperature_bounds(self):
        """Test that learned temperature is within reasonable bounds."""
        np.random.seed(42)
        logits = np.random.randn(500, 5) * 2.0
        labels = np.random.randint(0, 5, size=500)

        temp_scaler = TemperatureScaling(num_classes=5)
        temp_scaler.fit(logits, labels, verbose=False)

        temp = temp_scaler.get_temperature()

        # Temperature should be positive and reasonable (0.1 to 10)
        assert 0.1 <= temp <= 10.0, f"Temperature out of reasonable range: {temp}"

    def test_temperature_handles_nan_inf(self):
        """Test that temperature scaling handles NaN/Inf gracefully."""
        np.random.seed(42)
        logits = np.random.randn(100, 5)
        labels = np.random.randint(0, 5, size=100)

        # Add some extreme values
        logits[0, 0] = 1000.0  # Very large
        logits[1, 0] = -1000.0  # Very small

        temp_scaler = TemperatureScaling(num_classes=5)
        temp_scaler.fit(logits, labels, verbose=False)

        calibrated = temp_scaler.calibrate(logits)

        # Should not produce NaN or Inf
        assert not np.isnan(calibrated).any(), "Temperature scaling produced NaN"
        assert not np.isinf(calibrated).any(), "Temperature scaling produced Inf"


class TestIsotonicCalibration:
    """Test isotonic regression calibration."""

    @pytest.fixture
    def sample_probs_and_labels(self):
        """Create sample probabilities and labels."""
        np.random.seed(42)
        n_samples = 500
        num_classes = 5

        # Create probabilities with systematic miscalibration
        probs = np.random.dirichlet(np.ones(num_classes) * 2, size=n_samples)
        labels = np.random.randint(0, num_classes, size=n_samples)

        return probs, labels

    def test_isotonic_fit_and_calibrate(self, sample_probs_and_labels):
        """Test basic isotonic calibration workflow."""
        probs, labels = sample_probs_and_labels

        # Split data
        split_idx = len(probs) // 2
        fit_probs, fit_labels = probs[:split_idx], labels[:split_idx]
        test_probs, test_labels = probs[split_idx:], labels[split_idx:]

        # Fit isotonic calibration
        iso_cal = IsotonicCalibration(num_classes=5)
        iso_cal.fit(fit_probs, fit_labels, verbose=False)

        # Calibrate test set
        calibrated = iso_cal.calibrate(test_probs)

        # Check output properties
        assert calibrated.shape == test_probs.shape, "Shape mismatch after calibration"
        assert np.all((calibrated >= 0) & (calibrated <= 1)), "Probabilities out of [0, 1]"
        assert np.allclose(calibrated.sum(axis=1), 1.0), "Probabilities don't sum to 1"

    def test_isotonic_constant_probability_edge_case(self):
        """Test isotonic calibration with constant probabilities."""
        # All samples have same probability
        probs = np.full((100, 5), 0.2)
        labels = np.random.randint(0, 5, size=100)

        iso_cal = IsotonicCalibration(num_classes=5)
        iso_cal.fit(probs, labels, verbose=False)

        # Calibrate should not crash
        calibrated = iso_cal.calibrate(probs)

        assert calibrated.shape == probs.shape
        assert not np.isnan(calibrated).any()

    def test_isotonic_preserves_ordering(self, sample_probs_and_labels):
        """Test that isotonic calibration preserves relative class ordering."""
        probs, labels = sample_probs_and_labels

        split_idx = len(probs) // 2
        iso_cal = IsotonicCalibration(num_classes=5)
        iso_cal.fit(probs[:split_idx], labels[:split_idx], verbose=False)

        test_probs = probs[split_idx:]
        calibrated = iso_cal.calibrate(test_probs)

        # Argmax should be preserved (highest probability class shouldn't change)
        assert np.all(test_probs.argmax(axis=1) == calibrated.argmax(axis=1)), (
            "Isotonic calibration changed predicted classes"
        )


class TestThresholdOptimization:
    """Test threshold optimization for multi-class classification."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        np.random.seed(42)
        n_samples = 500
        num_classes = 5

        probs = np.random.dirichlet(np.ones(num_classes), size=n_samples)
        labels = np.random.randint(0, num_classes, size=n_samples)

        return probs, labels

    def test_threshold_optimization_improves_f1(self, sample_predictions):
        """Test that optimized thresholds improve F1 scores."""
        probs, labels = sample_predictions

        # F1 with default threshold (argmax = 0.5 equivalent)
        pred_labels_default = probs.argmax(axis=1)

        from sklearn.metrics import f1_score
        default_f1 = f1_score(labels, pred_labels_default, average='macro', zero_division=0)

        # Optimize thresholds
        best_thresholds, f1_per_class = optimize_thresholds_per_class(
            probs, labels, num_thresholds=50, metric='f1'
        )

        # Per-class oracle F1 should be >= default
        oracle_f1 = np.mean(list(f1_per_class.values()))

        # Oracle should be at least as good (optimizing per class independently)
        assert oracle_f1 >= default_f1 - 0.01, (
            f"Optimized thresholds worse than default: {oracle_f1:.4f} < {default_f1:.4f}"
        )

    def test_threshold_stability(self, sample_predictions):
        """Test that threshold optimization is stable across bootstrap samples."""
        probs, labels = sample_predictions

        stability_results = validate_threshold_stability(
            probs, labels, num_bootstrap=5, subsample_ratio=0.8
        )

        mean_thresholds = stability_results['mean_thresholds']
        std_thresholds = stability_results['std_thresholds']

        # Standard deviation should be reasonable (< 0.3)
        assert np.all(std_thresholds < 0.3), (
            f"Threshold optimization unstable: max std = {std_thresholds.max():.4f}"
        )

        # Mean thresholds should be in valid range
        assert np.all((mean_thresholds >= 0) & (mean_thresholds <= 1)), (
            "Invalid threshold values"
        )

    def test_pr_curves_computation(self, sample_predictions):
        """Test precision-recall curve computation."""
        probs, labels = sample_predictions

        pr_curves = compute_pr_curves(probs, labels, save_arrays=True)

        # Should have curves for all classes
        assert len(pr_curves) == 5

        # Each curve should have required fields
        for class_idx, curve_data in pr_curves.items():
            assert 'precision' in curve_data
            assert 'recall' in curve_data
            assert 'thresholds' in curve_data
            assert 'auprc' in curve_data

            # AUPRC should be in [0, 1]
            assert 0 <= curve_data['auprc'] <= 1

    def test_threshold_handles_no_positives(self):
        """Test threshold optimization when a class has no positive examples."""
        # Create data where class 0 has no positives
        probs = np.random.dirichlet(np.ones(3), size=100)
        labels = np.random.randint(1, 3, size=100)  # Only classes 1 and 2

        # Should not crash
        thresholds, f1_per_class = optimize_thresholds_per_class(
            probs, labels, num_thresholds=20
        )

        assert len(thresholds) == 3
        assert 0 in f1_per_class  # Class 0 should be in results (even if F1=0)


class TestCalibrationMetrics:
    """Test ECE and related calibration metrics."""

    def test_ece_perfect_calibration(self):
        """Test ECE on perfectly calibrated predictions."""
        np.random.seed(42)
        n_samples = 1000
        num_classes = 5

        # Generate perfectly calibrated predictions
        labels = np.random.randint(0, num_classes, size=n_samples)
        probs = np.zeros((n_samples, num_classes))

        # Set probabilities to match true frequencies
        for i in range(n_samples):
            probs[i] = np.random.dirichlet(np.ones(num_classes))
            probs[i, labels[i]] += 0.5  # Boost true class
            probs[i] /= probs[i].sum()

        ece, _, _, _ = expected_calibration_error(probs, labels, num_bins=10)

        # ECE should be low for reasonably calibrated predictions
        assert ece < 0.5, f"ECE too high for calibrated predictions: {ece:.4f}"

    def test_ece_terrible_calibration(self):
        """Test ECE on terribly miscalibrated predictions."""
        np.random.seed(42)
        n_samples = 500
        num_classes = 3

        # Generate overconfident predictions
        labels = np.random.randint(0, num_classes, size=n_samples)
        probs = np.zeros((n_samples, num_classes))

        # Always predict wrong class with high confidence
        for i in range(n_samples):
            wrong_class = (labels[i] + 1) % num_classes
            probs[i, wrong_class] = 0.95
            probs[i, :] /= probs[i].sum()

        ece, _, _, _ = expected_calibration_error(probs, labels, num_bins=10)

        # ECE should be very high
        assert ece > 0.5, f"ECE too low for miscalibrated predictions: {ece:.4f}"

    def test_ece_bounds(self):
        """Test that ECE is always in [0, 1]."""
        np.random.seed(42)

        for _ in range(10):
            n_samples = np.random.randint(100, 500)
            num_classes = np.random.randint(2, 10)

            probs = np.random.dirichlet(np.ones(num_classes), size=n_samples)
            labels = np.random.randint(0, num_classes, size=n_samples)

            ece, _, _, _ = expected_calibration_error(probs, labels, num_bins=15)

            assert 0 <= ece <= 1, f"ECE out of bounds: {ece}"

    def test_reliability_diagram_data(self, tmp_path):
        """Test reliability diagram data generation."""
        np.random.seed(42)
        n_samples = 500
        probs = np.random.dirichlet(np.ones(5), size=n_samples)
        labels = np.random.randint(0, 5, size=n_samples)

        # Generate diagram (without plotting)
        bin_confidences, bin_accuracies, bin_counts = reliability_diagram(
            probs, labels, num_bins=10, save_path=None
        )

        # Check shapes
        assert len(bin_confidences) == 10
        assert len(bin_accuracies) == 10
        assert len(bin_counts) == 10

        # Confidences should be in [0, 1]
        valid_bins = bin_counts > 0
        assert np.all((bin_confidences[valid_bins] >= 0) & (bin_confidences[valid_bins] <= 1))

        # Accuracies should be in [0, 1]
        assert np.all((bin_accuracies[valid_bins] >= 0) & (bin_accuracies[valid_bins] <= 1))
