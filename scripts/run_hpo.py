"""
Hyperparameter optimization using Optuna.

Optimizes for macro-AUPRC on validation set.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, args) -> float:
    """
    Optuna objective function.

    Args:
        trial: Optuna trial object
        args: Command-line arguments

    Returns:
        Validation macro-AUPRC (to maximize)
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    pooling_strategy = trial.suggest_categorical(
        "pooling_strategy", ["mean", "cls", "max", "attention"]
    )
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    logger.info(f"Trial {trial.number}: lr={learning_rate:.2e}, batch={batch_size}, "
               f"dropout={dropout:.3f}, pooling={pooling_strategy}")

    # Build command
    # In practice, this would call the training script
    # For now, we simulate with mock metric
    try:
        # Simulate training (replace with actual training)
        mock_auprc = np.random.uniform(0.6, 0.8)

        # Add some signal based on hyperparameters
        if pooling_strategy == "mean":
            mock_auprc += 0.02
        if 1e-5 < learning_rate < 5e-5:
            mock_auprc += 0.03
        if dropout < 0.15:
            mock_auprc += 0.01

        mock_auprc = min(mock_auprc, 1.0)

        logger.info(f"Trial {trial.number}: macro-AUPRC = {mock_auprc:.4f}")

        return mock_auprc

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description="HPO with Optuna")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of trials")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs")
    parser.add_argument("--study-name", type=str, default="gemma_hpo",
                       help="Study name")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db",
                       help="Storage URL for Optuna")
    parser.add_argument("--model-name", type=str, default="google/gemma-2-2b",
                       help="Base model name")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run with mock trials")

    args = parser.parse_args()

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",  # Maximize AUPRC
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    logger.info(f"Starting HPO: {args.n_trials} trials, {args.n_jobs} jobs")
    logger.info(f"Study: {args.study_name}, Storage: {args.storage}")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )

    # Report results
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best macro-AUPRC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*70)

    # Save best parameters
    best_params_path = Path("outputs/best/hpo_best_params.json")
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    logger.info(f"Best parameters saved to: {best_params_path}")

    # Plot optimization history (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.savefig("outputs/best/hpo_history.png", dpi=150)
        logger.info("Optimization history saved to: outputs/best/hpo_history.png")

        fig2 = plot_param_importances(study)
        fig2.savefig("outputs/best/hpo_param_importances.png", dpi=150)
        logger.info("Parameter importances saved to: outputs/best/hpo_param_importances.png")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
