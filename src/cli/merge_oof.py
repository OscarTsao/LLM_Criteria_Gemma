"""
Merge OOF predictions from multiple folds for comprehensive evaluation.

Concatenates OOF artifacts from all folds and computes aggregated metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.calibration import expected_calibration_error, reliability_diagram
from src.calibration.threshold_optimization import (
    optimize_thresholds_per_class,
    compute_pr_curves,
    compute_macro_auprc,
    compute_coverage_risk_curve
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_all_folds(experiment_dir: Path, num_folds: int) -> Dict[str, np.ndarray]:
    """
    Load and concatenate OOF predictions from all folds.

    Args:
        experiment_dir: Path to experiment directory containing fold_X subdirs
        num_folds: Number of folds to load

    Returns:
        Dict with concatenated arrays
    """
    all_probs = []
    all_labels = []
    all_logits = []
    all_ids = []
    has_logits = True
    has_ids = True

    for fold_idx in range(num_folds):
        fold_dir = experiment_dir / f"fold_{fold_idx}"

        if not fold_dir.exists():
            logger.warning(f"Fold {fold_idx} not found at {fold_dir}, skipping")
            continue

        # Load artifacts
        probs_file = fold_dir / "oof_probs.npy"
        labels_file = fold_dir / "oof_labels.npy"
        logits_file = fold_dir / "oof_logits.npy"
        ids_file = fold_dir / "ids.csv"

        if not probs_file.exists() or not labels_file.exists():
            logger.warning(f"Missing required files in {fold_dir}, skipping")
            continue

        probs = np.load(probs_file)
        labels = np.load(labels_file)

        all_probs.append(probs)
        all_labels.append(labels)

        if logits_file.exists() and has_logits:
            all_logits.append(np.load(logits_file))
        else:
            has_logits = False

        if ids_file.exists() and has_ids:
            all_ids.append(pd.read_csv(ids_file))
        else:
            has_ids = False

        logger.info(f"Loaded fold {fold_idx}: {len(probs)} samples")

    # Concatenate
    merged = {
        'probs': np.concatenate(all_probs, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
    }

    if has_logits:
        merged['logits'] = np.concatenate(all_logits, axis=0)

    if has_ids:
        merged['ids'] = pd.concat(all_ids, ignore_index=True)

    logger.info(f"Merged {len(all_probs)} folds")
    logger.info(f"Total samples: {len(merged['probs'])}")

    return merged


def validate_no_overlap(experiment_dir: Path, num_folds: int) -> bool:
    """
    Validate that no sample appears in multiple folds.

    Args:
        experiment_dir: Path to experiment directory
        num_folds: Number of folds

    Returns:
        True if no overlap detected
    """
    all_ids = set()
    overlap_detected = False

    for fold_idx in range(num_folds):
        fold_dir = experiment_dir / f"fold_{fold_idx}"
        ids_file = fold_dir / "ids.csv"

        if not ids_file.exists():
            logger.warning(f"No ids.csv in {fold_dir}, skipping overlap check")
            continue

        fold_ids = pd.read_csv(ids_file)
        if 'id' in fold_ids.columns:
            fold_id_set = set(fold_ids['id'].values)

            # Check for overlap
            overlap = all_ids.intersection(fold_id_set)
            if overlap:
                logger.error(f"Fold {fold_idx}: {len(overlap)} overlapping IDs detected!")
                logger.error(f"  Example overlaps: {list(overlap)[:5]}")
                overlap_detected = True

            all_ids.update(fold_id_set)

    if not overlap_detected:
        logger.info("✓ No overlap detected between folds")

    return not overlap_detected


def compute_global_metrics(
    merged: Dict[str, np.ndarray],
    output_dir: Path,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive global metrics on merged OOF predictions.

    Args:
        merged: Dict with probs, labels, etc.
        output_dir: Where to save outputs
        class_names: Optional class names for reporting

    Returns:
        Dict of metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    probs = merged['probs']
    labels = merged['labels']
    num_classes = probs.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    metrics = {}

    # === 1. Basic Metrics ===
    pred_labels = probs.argmax(axis=1)
    metrics['accuracy'] = accuracy_score(labels, pred_labels)
    metrics['macro_f1'] = f1_score(labels, pred_labels, average='macro', zero_division=0)

    # === 2. Calibration ===
    ece, _, _, _ = expected_calibration_error(probs, labels, num_bins=15)
    metrics['ece'] = ece

    reliability_diagram(
        probs, labels, num_bins=15,
        save_path=output_dir / "reliability_global.png"
    )

    # === 3. Threshold Optimization ===
    logger.info("Optimizing global thresholds...")
    best_thresholds, f1_per_class = optimize_thresholds_per_class(
        probs, labels, num_thresholds=100, metric='f1'
    )

    metrics['best_thresholds'] = best_thresholds.tolist()
    metrics['f1_per_class_oracle'] = f1_per_class

    # Global F1 with default threshold (0.5)
    metrics['macro_f1_global_threshold_0.5'] = metrics['macro_f1']

    # Per-class oracle F1 (each class uses its own best threshold)
    metrics['macro_f1_perclass_oracle'] = np.mean(list(f1_per_class.values()))

    # === 4. AUPRC ===
    pr_curves = compute_pr_curves(probs, labels, save_arrays=True)
    metrics['macro_auprc'] = compute_macro_auprc(pr_curves)
    metrics['auprc_per_class'] = {
        class_names[i]: pr_curves[i]['auprc'] for i in range(num_classes)
    }

    # Plot PR curves
    plot_pr_curves_with_names(pr_curves, class_names, output_dir / "pr_curves_global.png")

    # === 5. Coverage-Risk ===
    coverages, risks, thresholds = compute_coverage_risk_curve(probs, labels)
    np.savez(
        output_dir / "coverage_risk_global.npz",
        coverages=coverages,
        risks=risks,
        thresholds=thresholds
    )

    plot_coverage_risk(coverages, risks, output_dir / "coverage_risk_global.png")

    # === 6. Confusion Matrix ===
    cm = confusion_matrix(labels, pred_labels)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    # === 7. Classification Report ===
    report = classification_report(
        labels, pred_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report

    # === 8. Save Thresholds for Production ===
    threshold_config = {
        'thresholds': {
            class_names[i]: float(best_thresholds[i])
            for i in range(num_classes)
        },
        'default_threshold': 0.5,
        'num_classes': num_classes,
        'class_names': class_names
    }

    with open(output_dir / "thresholds.json", 'w') as f:
        json.dump(threshold_config, f, indent=2)

    # Save all metrics
    with open(output_dir / "global_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Global evaluation complete. Results saved to {output_dir}")

    return metrics


def plot_pr_curves_with_names(pr_curves: Dict, class_names: List[str], save_path: Path):
    """Plot PR curves with class names."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for class_idx, curve_data in pr_curves.items():
        if curve_data['recall'] is not None:
            ax.plot(
                curve_data['recall'],
                curve_data['precision'],
                label=f"{class_names[class_idx]} (AUPRC={curve_data['auprc']:.3f})",
                linewidth=2
            )

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (All Classes)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_coverage_risk(coverages: np.ndarray, risks: np.ndarray, save_path: Path):
    """Plot coverage-risk curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(coverages, risks, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Coverage (Fraction Predicted)', fontsize=12)
    ax.set_ylabel('Risk (Error Rate)', fontsize=12)
    ax.set_title('Coverage-Risk Curve', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Merge and evaluate OOF predictions from all folds")
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Path to experiment directory containing fold_X subdirs')
    parser.add_argument('--num-folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: experiment_dir/merged_oof)')
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                       help='Class names for reporting')
    parser.add_argument('--validate-no-overlap', action='store_true',
                       help='Validate no sample overlap between folds')

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    if args.output_dir is None:
        output_dir = experiment_dir / "merged_oof"
    else:
        output_dir = Path(args.output_dir)

    # Validate no overlap (if requested)
    if args.validate_no_overlap:
        validate_no_overlap(experiment_dir, args.num_folds)

    # Load and merge folds
    merged = load_all_folds(experiment_dir, args.num_folds)

    # Compute global metrics
    metrics = compute_global_metrics(merged, output_dir, args.class_names)

    # Print summary
    print("\n" + "="*70)
    print("GLOBAL OOF EVALUATION SUMMARY")
    print("="*70)
    print(f"Total samples: {len(merged['labels'])}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro AUPRC: {metrics['macro_auprc']:.4f}")
    print(f"Macro F1 (global threshold 0.5): {metrics['macro_f1_global_threshold_0.5']:.4f}")
    print(f"Macro F1 (per-class oracle): {metrics['macro_f1_perclass_oracle']:.4f}")
    print(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")
    print("\nPer-class AUPRC:")
    for class_name, auprc in metrics['auprc_per_class'].items():
        print(f"  {class_name}: {auprc:.4f}")
    print("\nThresholds saved to: " + str(output_dir / "thresholds.json"))
    print("="*70)


if __name__ == '__main__':
    main()
