"""
Comprehensive evaluation script for OOF predictions.

Loads saved OOF artifacts from training and computes:
- Calibration metrics (ECE)
- Per-class threshold optimization
- Precision-recall curves and AUPRC
- Coverage-risk curves
- Reliability diagrams
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.calibration import (
    TemperatureScaling,
    IsotonicCalibration,
    expected_calibration_error,
    reliability_diagram
)
from src.calibration.threshold_optimization import (
    optimize_thresholds_per_class,
    compute_pr_curves,
    compute_macro_auprc,
    compute_coverage_risk_curve
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")


def load_oof_artifacts(fold_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load OOF artifacts from a fold directory.

    Expected files:
    - oof_probs.npy: (N, num_classes) probabilities
    - oof_labels.npy: (N,) labels
    - oof_logits.npy: (N, num_classes) pre-softmax logits (optional)
    - ids.csv: Sample IDs (optional)

    Args:
        fold_dir: Path to fold output directory

    Returns:
        Dict with loaded arrays
    """
    artifacts = {}

    # Required files
    probs_file = fold_dir / "oof_probs.npy"
    labels_file = fold_dir / "oof_labels.npy"

    if not probs_file.exists():
        raise FileNotFoundError(f"Missing {probs_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Missing {labels_file}")

    artifacts['probs'] = np.load(probs_file)
    artifacts['labels'] = np.load(labels_file)

    # Optional files
    logits_file = fold_dir / "oof_logits.npy"
    if logits_file.exists():
        artifacts['logits'] = np.load(logits_file)

    ids_file = fold_dir / "ids.csv"
    if ids_file.exists():
        import pandas as pd
        artifacts['ids'] = pd.read_csv(ids_file)

    logger.info(f"Loaded OOF artifacts from {fold_dir}")
    logger.info(f"  Probs shape: {artifacts['probs'].shape}")
    logger.info(f"  Labels shape: {artifacts['labels'].shape}")

    return artifacts


def evaluate_fold(
    fold_dir: Path,
    output_dir: Path,
    calibration_method: str = 'temperature',
    num_bins: int = 15
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a single fold.

    Args:
        fold_dir: Path to fold directory with OOF artifacts
        output_dir: Path to save evaluation outputs
        calibration_method: 'temperature', 'isotonic', or 'none'
        num_bins: Number of bins for ECE calculation

    Returns:
        Dict of metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOF predictions
    artifacts = load_oof_artifacts(fold_dir)
    probs = artifacts['probs']
    labels = artifacts['labels']
    logits = artifacts.get('logits', None)

    num_classes = probs.shape[1]
    metrics = {'fold': fold_dir.name}

    # === 1. Uncalibrated Metrics ===
    logger.info("Computing uncalibrated metrics...")

    pred_labels = probs.argmax(axis=1)
    metrics['uncalibrated'] = {
        'accuracy': accuracy_score(labels, pred_labels),
        'macro_f1': f1_score(labels, pred_labels, average='macro', zero_division=0),
        'macro_precision': precision_score(labels, pred_labels, average='macro', zero_division=0),
        'macro_recall': recall_score(labels, pred_labels, average='macro', zero_division=0),
    }

    # ECE before calibration
    ece_uncal, _, _, _ = expected_calibration_error(probs, labels, num_bins)
    metrics['uncalibrated']['ece'] = ece_uncal

    # Reliability diagram (uncalibrated)
    reliability_diagram(
        probs, labels, num_bins,
        save_path=output_dir / "reliability_uncalibrated.png"
    )

    # === 2. Calibration ===
    calibrated_probs = probs.copy()

    if calibration_method == 'temperature' and logits is not None:
        logger.info("Applying temperature scaling...")
        temp_scaler = TemperatureScaling(num_classes=num_classes)
        # Use part of data for fitting (e.g., 50%)
        split_idx = len(logits) // 2
        temp_scaler.fit(logits[:split_idx], labels[:split_idx], verbose=True)
        calibrated_probs = temp_scaler.calibrate(logits)
        metrics['temperature'] = temp_scaler.get_temperature()

    elif calibration_method == 'isotonic':
        logger.info("Applying isotonic regression...")
        iso_cal = IsotonicCalibration(num_classes=num_classes)
        split_idx = len(probs) // 2
        iso_cal.fit(probs[:split_idx], labels[:split_idx], verbose=True)
        calibrated_probs = iso_cal.calibrate(probs)

    # ECE after calibration
    if calibration_method != 'none':
        ece_cal, _, _, _ = expected_calibration_error(calibrated_probs, labels, num_bins)
        metrics['calibrated'] = {'ece': ece_cal}

        # Reliability diagram (calibrated)
        reliability_diagram(
            calibrated_probs, labels, num_bins,
            save_path=output_dir / "reliability_calibrated.png"
        )

        logger.info(f"ECE: {ece_uncal:.4f} -> {ece_cal:.4f}")

    # === 3. Threshold Optimization ===
    logger.info("Optimizing per-class thresholds...")

    best_thresholds, f1_at_best = optimize_thresholds_per_class(
        calibrated_probs, labels, num_thresholds=100, metric='f1'
    )

    metrics['thresholds'] = best_thresholds.tolist()
    metrics['f1_per_class_at_best_threshold'] = f1_at_best

    # Apply thresholds for global F1
    one_hot = np.zeros_like(calibrated_probs)
    one_hot[np.arange(len(labels)), labels] = 1

    threshold_preds = np.zeros_like(calibrated_probs)
    for class_idx in range(num_classes):
        threshold_preds[:, class_idx] = (calibrated_probs[:, class_idx] >= best_thresholds[class_idx]).astype(int)

    # For multi-class, take argmax of threshold predictions
    final_preds = threshold_preds.argmax(axis=1)
    metrics['calibrated']['f1_with_optimized_thresholds'] = f1_score(labels, final_preds, average='macro', zero_division=0)

    # === 4. PR Curves and AUPRC ===
    logger.info("Computing PR curves...")

    pr_curves = compute_pr_curves(calibrated_probs, labels, save_arrays=True)
    macro_auprc = compute_macro_auprc(pr_curves)
    metrics['macro_auprc'] = macro_auprc

    # Save per-class AUPRC
    metrics['auprc_per_class'] = {
        class_idx: pr_curves[class_idx]['auprc']
        for class_idx in range(num_classes)
    }

    # Plot PR curves
    plot_pr_curves(pr_curves, save_path=output_dir / "pr_curves.png")

    # === 5. Coverage-Risk Curve ===
    logger.info("Computing coverage-risk curve...")

    coverages, risks, thresholds = compute_coverage_risk_curve(calibrated_probs, labels)

    plot_coverage_risk(coverages, risks, save_path=output_dir / "coverage_risk.png")

    # Save curve data
    np.savez(
        output_dir / "coverage_risk_data.npz",
        coverages=coverages,
        risks=risks,
        thresholds=thresholds
    )

    # === 6. Classification Report ===
    report = classification_report(labels, pred_labels, output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"  Macro AUPRC: {macro_auprc:.4f}")
    logger.info(f"  Macro F1: {metrics['uncalibrated']['macro_f1']:.4f}")

    return metrics


def plot_pr_curves(pr_curves: Dict, save_path: Path):
    """Plot precision-recall curves for all classes."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for class_idx, curve_data in pr_curves.items():
        if curve_data['recall'] is not None:
            ax.plot(
                curve_data['recall'],
                curve_data['precision'],
                label=f"Class {class_idx} (AUPRC={curve_data['auprc']:.3f})",
                linewidth=2
            )

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (Per Class)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_coverage_risk(coverages: np.ndarray, risks: np.ndarray, save_path: Path):
    """Plot coverage-risk curve for selective prediction."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, risks, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Coverage (Fraction Predicted)', fontsize=12)
    ax.set_ylabel('Risk (Error Rate)', fontsize=12)
    ax.set_title('Coverage-Risk Curve (Selective Prediction)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOF predictions")
    parser.add_argument('--fold-dir', type=str, required=True,
                       help='Path to fold directory (e.g., outputs/exp1/fold_0)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results')
    parser.add_argument('--calibration', type=str, default='temperature',
                       choices=['temperature', 'isotonic', 'none'],
                       help='Calibration method')
    parser.add_argument('--num-bins', type=int, default=15,
                       help='Number of bins for ECE calculation')

    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    if args.output_dir is None:
        output_dir = fold_dir / "evaluation"
    else:
        output_dir = Path(args.output_dir)

    metrics = evaluate_fold(
        fold_dir=fold_dir,
        output_dir=output_dir,
        calibration_method=args.calibration,
        num_bins=args.num_bins
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Fold: {metrics['fold']}")
    print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
    print(f"Macro F1 (uncalibrated): {metrics['uncalibrated']['macro_f1']:.4f}")
    print(f"ECE (uncalibrated): {metrics['uncalibrated']['ece']:.4f}")
    if 'calibrated' in metrics:
        print(f"ECE (calibrated): {metrics['calibrated']['ece']:.4f}")
        print(f"Macro F1 (with optimized thresholds): {metrics['calibrated']['f1_with_optimized_thresholds']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
