"""
Terminal Visualization Utilities for Training and Inference

Provides beautiful terminal-based visualizations using Rich library:
- Training progress with metrics
- Real-time loss curves
- Confusion matrices
- Prediction results
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not available. Install with: pip install rich")

try:
    import plotext as plt
    PLOTEXT_AVAILABLE = True
except ImportError:
    PLOTEXT_AVAILABLE = False


console = Console() if RICH_AVAILABLE else None


class TrainingVisualizer:
    """Visualize training progress in terminal."""

    def __init__(self, num_epochs: int, num_folds: int = 1):
        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.history = []

    def print_header(self, fold: int = None):
        """Print training header."""
        if not RICH_AVAILABLE:
            if fold is not None:
                print(f"\n{'=' * 60}\nFold {fold}/{self.num_folds}\n{'=' * 60}")
            else:
                print(f"\n{'=' * 60}\nTraining Started\n{'=' * 60}")
            return

        if fold is not None:
            title = f"Fold {fold}/{self.num_folds}"
        else:
            title = "Training Started"

        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]\n"
            f"Epochs: {self.num_epochs}",
            box=box.DOUBLE,
            expand=False
        ))

    def print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """Print epoch summary with metrics."""
        if not RICH_AVAILABLE:
            print(f"\nEpoch {epoch}/{self.num_epochs}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            return

        # Create metrics table
        table = Table(title=f"Epoch {epoch}/{self.num_epochs}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
            else:
                table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)

        # Add to history
        self.history.append({'epoch': epoch, **metrics})

    def plot_training_curves(self, save_path: Optional[Path] = None):
        """Plot training curves in terminal."""
        if not PLOTEXT_AVAILABLE or not self.history:
            return

        epochs = [h['epoch'] for h in self.history]

        # Plot loss curves
        if 'train_loss' in self.history[0]:
            plt.clf()
            plt.title("Training & Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            train_loss = [h['train_loss'] for h in self.history]
            plt.plot(epochs, train_loss, label="Train Loss", marker="braille")

            if 'val_loss' in self.history[0]:
                val_loss = [h['val_loss'] for h in self.history]
                plt.plot(epochs, val_loss, label="Val Loss", marker="braille")

            plt.show()

        # Plot metrics
        if 'val_f1' in self.history[0]:
            plt.clf()
            plt.title("Validation Metrics")
            plt.xlabel("Epoch")
            plt.ylabel("Score")

            val_f1 = [h['val_f1'] for h in self.history]
            plt.plot(epochs, val_f1, label="F1 Score", marker="braille")

            if 'val_accuracy' in self.history[0]:
                val_acc = [h['val_accuracy'] for h in self.history]
                plt.plot(epochs, val_acc, label="Accuracy", marker="braille")

            plt.show()

    def print_fold_summary(self, fold_results: Dict[str, Any]):
        """Print summary for completed fold."""
        if not RICH_AVAILABLE:
            print(f"\nFold {fold_results.get('fold', '?')} Complete:")
            print(f"  Best Val F1: {fold_results.get('best_val_f1', 0):.4f}")
            print(f"  Best Val AUC: {fold_results.get('best_val_auc', 0):.4f}")
            return

        console.print(Panel(
            f"[bold green]Fold {fold_results.get('fold', '?')} Complete[/bold green]\n"
            f"Best Val F1:  {fold_results.get('best_val_f1', 0):.4f}\n"
            f"Best Val AUC: {fold_results.get('best_val_auc', 0):.4f}",
            box=box.DOUBLE,
            style="green"
        ))

    def print_cv_summary(self, fold_results: List[Dict[str, Any]]):
        """Print cross-validation summary."""
        if not RICH_AVAILABLE:
            print("\n" + "=" * 60)
            print("Cross-Validation Summary")
            print("=" * 60)
            for result in fold_results:
                print(f"Fold {result['fold']}: F1={result['best_val_f1']:.4f}, AUC={result['best_val_auc']:.4f}")

            f1_scores = [r['best_val_f1'] for r in fold_results]
            auc_scores = [r['best_val_auc'] for r in fold_results]
            print(f"\nMean F1:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
            return

        # Create summary table
        table = Table(
            title="[bold]Cross-Validation Summary[/bold]",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Fold", justify="center", style="cyan")
        table.add_column("F1 Score", justify="right", style="green")
        table.add_column("AUC Score", justify="right", style="blue")

        for result in fold_results:
            table.add_row(
                str(result['fold']),
                f"{result['best_val_f1']:.4f}",
                f"{result['best_val_auc']:.4f}"
            )

        # Add aggregate statistics
        f1_scores = [r['best_val_f1'] for r in fold_results]
        auc_scores = [r['best_val_auc'] for r in fold_results]

        table.add_section()
        table.add_row(
            "[bold]Mean ± Std[/bold]",
            f"[bold]{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}[/bold]",
            f"[bold]{np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}[/bold]"
        )

        console.print("\n")
        console.print(table)


class InferenceVisualizer:
    """Visualize inference results in terminal."""

    def print_prediction(self,
                        post: str,
                        criterion: str,
                        prediction: int,
                        probability: float,
                        true_label: Optional[int] = None):
        """Print single prediction with formatting."""
        if not RICH_AVAILABLE:
            print("\n" + "=" * 60)
            print(f"Post: {post[:100]}...")
            print(f"Criterion: {criterion[:100]}...")
            print(f"Prediction: {'Matched' if prediction == 1 else 'Unmatched'} ({probability:.2%})")
            if true_label is not None:
                correct = "✓" if prediction == true_label else "✗"
                print(f"True Label: {'Matched' if true_label == 1 else 'Unmatched'} {correct}")
            return

        # Create prediction panel
        prediction_text = Text()
        prediction_text.append("Prediction: ", style="bold")

        pred_label = "Matched" if prediction == 1 else "Unmatched"
        pred_style = "bold green" if prediction == 1 else "bold red"
        prediction_text.append(f"{pred_label}\n", style=pred_style)
        prediction_text.append(f"Confidence: {probability:.2%}\n", style="cyan")

        if true_label is not None:
            correct = prediction == true_label
            true_label_text = "Matched" if true_label == 1 else "Unmatched"

            prediction_text.append("True Label: ", style="bold")
            prediction_text.append(f"{true_label_text} ", style="yellow")
            prediction_text.append("✓" if correct else "✗",
                                  style="green" if correct else "red")

        console.print(Panel(
            f"[bold]Post:[/bold]\n{post[:200]}{'...' if len(post) > 200 else ''}\n\n"
            f"[bold]Criterion:[/bold]\n{criterion[:200]}{'...' if len(criterion) > 200 else ''}\n\n"
            f"{prediction_text}",
            box=box.ROUNDED,
            expand=False
        ))

    def print_batch_predictions(self,
                               predictions: List[Dict[str, Any]],
                               show_details: bool = False):
        """Print batch predictions as table."""
        if not RICH_AVAILABLE:
            print(f"\nBatch Predictions ({len(predictions)} samples):")
            correct = sum(1 for p in predictions if p.get('correct', False))
            print(f"Accuracy: {correct}/{len(predictions)} ({correct/len(predictions):.2%})")
            return

        table = Table(
            title=f"Batch Predictions ({len(predictions)} samples)",
            box=box.ROUNDED,
            show_header=True
        )

        table.add_column("#", justify="right", style="cyan")
        table.add_column("Prediction", justify="center")
        table.add_column("Confidence", justify="right", style="green")
        table.add_column("True Label", justify="center", style="yellow")
        table.add_column("Result", justify="center")

        if show_details:
            table.add_column("Post Preview", style="dim")

        for i, pred in enumerate(predictions):
            pred_label = "✓ Match" if pred['prediction'] == 1 else "✗ No Match"
            true_label = "✓ Match" if pred.get('true_label') == 1 else "✗ No Match"

            correct = pred.get('correct', False)
            result = "[green]✓[/green]" if correct else "[red]✗[/red]"

            row = [
                str(i + 1),
                pred_label,
                f"{pred['probability']:.2%}",
                true_label if pred.get('true_label') is not None else "—",
                result
            ]

            if show_details:
                post_preview = pred.get('post', '')[:50] + "..."
                row.append(post_preview)

            table.add_row(*row)

        console.print(table)

        # Print accuracy
        if any('correct' in p for p in predictions):
            correct = sum(1 for p in predictions if p.get('correct', False))
            accuracy = correct / len(predictions)

            console.print(f"\n[bold]Overall Accuracy:[/bold] {correct}/{len(predictions)} "
                        f"[green]({accuracy:.2%})[/green]")

    def print_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None):
        """Print confusion matrix."""
        if labels is None:
            labels = ["Unmatched", "Matched"]

        if not RICH_AVAILABLE:
            print("\nConfusion Matrix:")
            print(cm)
            return

        table = Table(title="Confusion Matrix", box=box.DOUBLE_EDGE)
        table.add_column("", style="bold")
        for label in labels:
            table.add_column(f"Pred {label}", justify="center")

        for i, label in enumerate(labels):
            row = [f"True {label}"]
            for j in range(len(labels)):
                value = cm[i, j]
                # Highlight diagonal (correct predictions)
                if i == j:
                    row.append(f"[bold green]{value}[/bold green]")
                else:
                    row.append(f"[red]{value}[/red]")
            table.add_row(*row)

        console.print(table)


def create_progress_bar(description: str = "Processing"):
    """Create a rich progress bar."""
    if not RICH_AVAILABLE:
        return None

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )


def print_model_info(model_name: str, num_params: int, config: Dict[str, Any]):
    """Print model information."""
    if not RICH_AVAILABLE:
        print(f"\nModel: {model_name}")
        print(f"Parameters: {num_params:,}")
        print("Configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        return

    info_text = f"[bold cyan]Model:[/bold cyan] {model_name}\n"
    info_text += f"[bold cyan]Parameters:[/bold cyan] {num_params:,}\n\n"
    info_text += "[bold]Configuration:[/bold]\n"

    for key, value in config.items():
        info_text += f"  {key}: {value}\n"

    console.print(Panel(info_text, title="Model Information", box=box.ROUNDED))


def print_dataset_info(num_samples: int, num_positive: int, num_negative: int):
    """Print dataset information."""
    if not RICH_AVAILABLE:
        print(f"\nDataset: {num_samples} samples")
        print(f"  Positive (matched): {num_positive}")
        print(f"  Negative (unmatched): {num_negative}")
        print(f"  Balance: {num_positive/num_samples:.1%} positive")
        return

    table = Table(title="Dataset Information", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Samples", f"{num_samples:,}")
    table.add_row("Positive (Matched)", f"{num_positive:,}")
    table.add_row("Negative (Unmatched)", f"{num_negative:,}")
    table.add_row("Positive Ratio", f"{num_positive/num_samples:.1%}")

    console.print(table)
