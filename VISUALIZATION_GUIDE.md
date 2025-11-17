# Terminal Visualization Guide

Beautiful terminal-based visualization for training and inference using Rich and Plotext.

## Overview

The project now includes comprehensive terminal visualization for:
- **Training Progress**: Real-time metrics, progress bars, and loss curves
- **Inference Results**: Beautiful formatted predictions and batch results
- **Cross-Validation**: Aggregate statistics and per-fold summaries

## Installation

### Install All Dependencies (Including Visualization)
```bash
make install
```

### Install Visualization Libraries Only
```bash
make install-viz
# or
pip install rich>=13.0.0 plotext>=5.2.0
```

---

## Training Visualization

### Features

✅ **Progress Bars** - Real-time training progress with tqdm
✅ **Metrics Tables** - Beautiful formatted epoch summaries
✅ **Terminal Plots** - Loss curves and metric plots in terminal
✅ **Fold Summaries** - Cross-validation results
✅ **Color-Coded Output** - Easy-to-read colored terminal output

### Example Output

```
╔═══════════════════════════════════════╗
║  Fold 1/5                              ║
║  Epochs: 20                            ║
╚═══════════════════════════════════════╝

     Epoch 1/20
┌──────────────┬────────┐
│ Metric       │  Value │
├──────────────┼────────┤
│ Train Loss   │ 0.6543 │
│ Val Loss     │ 0.5821 │
│ Val Accuracy │ 0.7234 │
│ Val F1       │ 0.7102 │
│ Val AUC      │ 0.8145 │
└──────────────┴────────┘

╔═══════════════════════════════════════╗
║  Fold 1 Complete                       ║
║  Best Val F1:  0.7521                  ║
║  Best Val AUC: 0.8342                  ║
╚═══════════════════════════════════════╝
```

### Usage

Training scripts automatically use visualization when Rich is installed.

---

## Inference Visualization

### Interactive Mode

Start an interactive prediction session:

```bash
# With specific checkpoint
make nli-predict-interactive CHECKPOINT=outputs/nli_full_5fold/fold_1_best.pt

# With latest model (automatic)
make nli-predict-best
```

**Features:**
- Enter posts and select criteria interactively
- See beautifully formatted predictions
- Confidence scores with color coding
- Support for custom criteria text

### Demo Mode

Run pre-defined examples:

```bash
# With specific checkpoint
make nli-predict-demo CHECKPOINT=outputs/nli_full_5fold/fold_1_best.pt

# With latest model (automatic)
make nli-demo-best
```

**Features:**
- 5 example predictions with ground truth labels
- Accuracy tracking
- Step-through interface

### Example Output

```
┌─────────────────────────────────────────────────────────┐
│ Post:                                                    │
│ I've been feeling really down lately. Everything feels  │
│ hopeless and I can't seem to find joy in anything...    │
│                                                          │
│ Criterion:                                               │
│ The patient exhibits a depressed mood most of the day,  │
│ nearly every day, as indicated by subjective report...  │
│                                                          │
│ Prediction: Matched                                     │
│ Confidence: 89.34%                                      │
│ True Label: Matched ✓                                   │
└─────────────────────────────────────────────────────────┘
```

### Batch Mode

Predict multiple posts from a file:

```bash
make nli-predict-batch \
    CHECKPOINT=outputs/nli_full_5fold/fold_1_best.pt \
    FILE=posts.txt \
    CRITERION=DEPRESSED_MOOD
```

**Output Table:**
```
     Batch Predictions (100 samples)
┌────┬─────────────┬────────────┬────────────┬────────┐
│ #  │ Prediction  │ Confidence │ True Label │ Result │
├────┼─────────────┼────────────┼────────────┼────────┤
│ 1  │ ✓ Match     │ 87.23%     │ ✓ Match    │   ✓    │
│ 2  │ ✗ No Match  │ 92.14%     │ ✗ No Match │   ✓    │
│ 3  │ ✓ Match     │ 78.45%     │ ✓ Match    │   ✓    │
└────┴─────────────┴────────────┴────────────┴────────┘

Overall Accuracy: 87/100 (87.00%)
```

---

## API Usage

### Training Visualization

```python
from src.utils.terminal_viz import TrainingVisualizer

# Initialize
viz = TrainingVisualizer(num_epochs=20, num_folds=5)

# Print header
viz.print_header(fold=1)

# Print epoch metrics
metrics = {
    'train_loss': 0.6543,
    'val_loss': 0.5821,
    'val_accuracy': 0.7234,
    'val_f1': 0.7102,
    'val_auc': 0.8145,
}
viz.print_epoch_summary(epoch=1, metrics=metrics)

# Plot training curves (if plotext available)
viz.plot_training_curves()

# Print fold summary
fold_result = {
    'fold': 1,
    'best_val_f1': 0.7521,
    'best_val_auc': 0.8342,
}
viz.print_fold_summary(fold_result)

# Print CV summary
viz.print_cv_summary(all_fold_results)
```

### Inference Visualization

```python
from src.utils.terminal_viz import InferenceVisualizer

viz = InferenceVisualizer()

# Print single prediction
viz.print_prediction(
    post="I feel tired all the time...",
    criterion="The patient reports fatigue...",
    prediction=1,  # matched
    probability=0.8934,
    true_label=1  # optional
)

# Print batch predictions
predictions = [
    {
        'prediction': 1,
        'probability': 0.87,
        'true_label': 1,
        'correct': True,
        'post': "Sample post..."
    },
    # ... more predictions
]
viz.print_batch_predictions(predictions, show_details=True)

# Print confusion matrix
import numpy as np
cm = np.array([[45, 5], [3, 47]])
viz.print_confusion_matrix(cm, labels=["Unmatched", "Matched"])
```

### Utility Functions

```python
from src.utils.terminal_viz import (
    create_progress_bar,
    print_model_info,
    print_dataset_info
)

# Progress bar
with create_progress_bar("Processing") as progress:
    task = progress.add_task("Loading data", total=100)
    for i in range(100):
        # Do work...
        progress.update(task, advance=1)

# Model info
print_model_info(
    model_name="google/gemma-2-2b",
    num_params=2_000_000_000,
    config={'pooling': 'mean', 'freeze': True}
)

# Dataset info
print_dataset_info(
    num_samples=3094,
    num_positive=1547,
    num_negative=1547
)
```

---

## Makefile Commands

### Visualization Setup
```bash
make install-viz          # Install visualization libraries
```

### Interactive Inference
```bash
make nli-predict-best     # Interactive with latest model
make nli-demo-best        # Demo with latest model

# Or specify checkpoint
make nli-predict-interactive CHECKPOINT=path/to/model.pt
make nli-predict-demo CHECKPOINT=path/to/model.pt
```

### Batch Inference
```bash
make nli-predict-batch \
    CHECKPOINT=path/to/model.pt \
    FILE=posts.txt \
    CRITERION=DEPRESSED_MOOD
```

---

## Terminal Requirements

### Supported Terminals

✅ **Linux/macOS**: Any modern terminal (supports Unicode and colors)
✅ **Windows**: Windows Terminal, PowerShell 7+, WSL
⚠️ **Legacy Windows CMD**: Limited color support

### Feature Availability

| Feature | Required Library | Fallback |
|---------|-----------------|----------|
| Colored output | `rich` | Plain text |
| Progress bars | `rich` or `tqdm` | Text updates |
| Tables | `rich` | Plain text |
| Terminal plots | `plotext` | Skipped |

### Graceful Degradation

If visualization libraries are not installed, the code gracefully falls back to plain text output:

```python
# Automatically detects if rich is available
if not RICH_AVAILABLE:
    print("Warning: 'rich' library not available")
    # Falls back to plain text output
```

---

## Examples

### Complete Training Workflow

```bash
# Install dependencies
make install-viz

# Train with visualization
make nli-train

# View results with visualization
make nli-show-results

# Run interactive predictions
make nli-predict-best
```

### Quick Demo

```bash
# After training, run demo
make nli-demo-best

# Or with specific model
make nli-predict-demo CHECKPOINT=outputs/nli_full_5fold/fold_1_best.pt
```

---

## Customization

### Colors

Customize colors in `src/utils/terminal_viz.py`:

```python
# Change styles in Rich panels
console.print(Panel(
    text,
    style="green",  # Change to: cyan, yellow, red, blue, etc.
    box=box.ROUNDED  # Change to: DOUBLE, MINIMAL, HEAVY, etc.
))
```

### Plot Settings

Customize plotext plots:

```python
import plotext as plt

plt.theme('dark')  # or 'clear', 'pro', 'windows'
plt.colorless()    # Disable colors
plt.plotsize(100, 30)  # Set plot size
```

---

## Troubleshooting

### Issue: Colors not showing
```bash
# Check terminal support
echo $TERM

# Force color support
export FORCE_COLOR=1
```

### Issue: Unicode characters broken
```bash
# Set UTF-8 encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### Issue: Rich not found
```bash
# Install visualization libraries
make install-viz

# Or manually
pip install rich plotext
```

### Issue: Plots not showing
```bash
# Install plotext
pip install plotext

# Or disable plots (optional)
# Plots are automatically skipped if not available
```

---

## Advanced Features

### Live Dashboard

Create a live updating dashboard during training:

```python
from rich.live import Live
from rich.table import Table

def generate_table(metrics):
    table = Table(title="Training Progress")
    # Add columns and rows...
    return table

with Live(generate_table(metrics), refresh_per_second=4) as live:
    for epoch in range(num_epochs):
        # Training...
        metrics = train_epoch()
        live.update(generate_table(metrics))
```

### Custom Layouts

Create complex layouts:

```python
from rich.layout import Layout
from rich.panel import Panel

layout = Layout()
layout.split_column(
    Layout(name="header"),
    Layout(name="body"),
    Layout(name="footer")
)

layout["header"].update(Panel("Training Progress"))
layout["body"].split_row(
    Layout(name="metrics"),
    Layout(name="plots")
)
```

---

## See Also

- **Rich Documentation**: https://rich.readthedocs.io/
- **Plotext Documentation**: https://github.com/piccolomo/plotext
- **README_NLI.md**: NLI task documentation
- **MAKEFILE_COMMANDS.md**: Complete command reference

---

## Summary

✅ **Beautiful terminal output** with Rich
✅ **Real-time progress tracking** with progress bars
✅ **Terminal plots** with Plotext
✅ **Interactive inference** with formatted predictions
✅ **Batch processing** with result tables
✅ **Graceful fallback** if libraries not available
✅ **Easy integration** via Makefile commands

**Get started:**
```bash
make install-viz
make nli-train
make nli-predict-best
```
