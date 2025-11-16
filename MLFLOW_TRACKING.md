# MLflow Experiment Tracking Guide

This guide explains how to use MLflow for comprehensive experiment tracking, model registry, and visualization in the Gemma NLI project.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [What Gets Tracked](#what-gets-tracked)
- [MLflow UI](#mlflow-ui)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Model Registry](#model-registry)
- [Comparing Experiments](#comparing-experiments)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

**MLflow** is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment. This project uses MLflow to track:

- Hyperparameters and configuration
- Training metrics (loss, F1, AUC, etc.)
- Hardware information
- Model artifacts and checkpoints
- Dataset metadata
- Per-fold results (for 5-fold CV)
- Aggregate cross-validation metrics

**Benefits:**
- ✅ Compare different training runs side-by-side
- ✅ Visualize metrics over time
- ✅ Track model versions and lineage
- ✅ Reproduce experiments exactly
- ✅ Share results with team members

---

## Quick Start

### 1. Install MLflow

MLflow is included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install separately:

```bash
pip install mlflow>=2.8.0
```

### 2. Train with MLflow Enabled

**5-Fold Cross-Validation (default: enabled)**

```bash
# MLflow is enabled by default
make nli-train

# Or with custom config
python src/training/train_nli_5fold.py experiment=nli_full_5fold
```

**Simple Training (requires flag)**

```bash
python src/training/train_nli_simple.py \
    --data_dir data/redsm5 \
    --mlflow \
    --epochs 10
```

### 3. View Results in MLflow UI

```bash
make mlflow-ui
```

Then open http://localhost:5000 in your browser.

---

## What Gets Tracked

### Hyperparameters

All configuration parameters are logged:

```yaml
model/name: google/gemma-2-2b
model/pooling_strategy: mean
model/freeze_encoder: true
training/batch_size: 8
training/learning_rate: 2e-5
training/num_epochs: 20
data/negative_ratio: 1.0
device/mixed_precision: true
...
```

### Hardware Information

Hardware specs are tagged for each run:

```yaml
gpu_name: NVIDIA RTX 4090
gpu_memory_gb: 24.0
has_gpu: true
supports_bfloat16: true
supports_tf32: true
```

### Dataset Metadata

Dataset statistics are logged:

```yaml
dataset/total_pairs: 3094
dataset/positive_pairs: 1547
dataset/negative_pairs: 1547
dataset/num_criteria: 10
```

### Training Metrics

**5-Fold CV Training** logs per-fold and aggregate metrics:

```
# Per-Fold Metrics (logged every epoch)
fold_1/train_loss: 0.4521
fold_1/val_loss: 0.3821
fold_1/val_f1: 0.7654
fold_1/val_auc: 0.8421
fold_2/train_loss: 0.4432
fold_2/val_loss: 0.3756
...

# Fold Summary (best metrics per fold)
cv_summary/fold_1_best_f1: 0.7654
cv_summary/fold_1_best_auc: 0.8421
cv_summary/fold_2_best_f1: 0.7801
...

# Aggregate Metrics (final results)
cv_aggregate/mean_f1: 0.7728
cv_aggregate/std_f1: 0.0123
cv_aggregate/median_f1: 0.7750
cv_aggregate/mean_auc: 0.8494
cv_aggregate/std_auc: 0.0089
```

**Simple Training** logs:

```
train_loss: (per epoch)
val_loss: (per epoch)
val_accuracy: (per epoch)
val_precision: (per epoch)
val_recall: (per epoch)
val_f1: (per epoch)
val_auc: (per epoch)

# Final test metrics
test_accuracy: 0.8234
test_f1: 0.7845
test_auc: 0.8521
best_val_f1: 0.7912
```

### Artifacts

All important files are logged as artifacts:

- **Configuration**: `config.yaml`
- **Model Checkpoints**: `fold_1_best.pt`, `fold_2_best.pt`, etc.
- **Training History**: `fold_1_history.json`
- **Aggregate Results**: `aggregate_results.json`
- **Dataset Info**: `nli_pairs.csv`, `nli_cv_folds_metadata.json`

---

## MLflow UI

### Launch the UI

```bash
make mlflow-ui
```

Or on a custom port:

```bash
make mlflow-ui-custom PORT=8080
```

### UI Features

**1. Experiments View**
- See all experiments in one place
- Filter by tags, parameters, metrics
- Sort runs by any metric

**2. Runs Comparison**
- Select multiple runs
- Compare metrics side-by-side
- View parameter differences
- Plot metric curves

**3. Run Details**
- View all parameters and metrics
- Download artifacts (models, configs, results)
- See hardware information
- Track experiment lineage

**4. Metric Charts**
- Plot training curves
- Compare metrics across runs
- Export charts as images

---

## Configuration

### Enable/Disable MLflow

**In config file** (`conf/config_nli.yaml`):

```yaml
mlflow:
  enabled: true  # Set to false to disable
  tracking_uri: ./mlruns
  experiment_name: gemma_nli_redsm5
  artifact_location: null
```

**From command line**:

```bash
# Disable MLflow
python src/training/train_nli_5fold.py mlflow.enabled=false

# Change experiment name
python src/training/train_nli_5fold.py mlflow.experiment_name=my_experiment
```

### Tracking URI Options

**Local file storage (default)**:
```yaml
tracking_uri: ./mlruns
```

**Remote MLflow server**:
```yaml
tracking_uri: http://mlflow-server:5000
```

**Database backend**:
```yaml
tracking_uri: postgresql://user:pass@localhost/mlflow
```

---

## Usage Examples

### Example 1: Compare Different Models

```bash
# Train with Gemma-2B
python src/training/train_nli_5fold.py \
    model.name=google/gemma-2-2b \
    output.experiment_name=gemma_2b_run1

# Train with Gemma-9B
python src/training/train_nli_5fold.py \
    model.name=google/gemma-2-9b \
    training.batch_size=4 \
    output.experiment_name=gemma_9b_run1

# View in MLflow UI
make mlflow-ui
```

In the UI:
1. Select both runs
2. Click "Compare"
3. View side-by-side metrics

### Example 2: Pooling Strategy Ablation

```bash
# Run ablation study
make nli-ablation-pooling

# View all runs
make mlflow-runs

# Launch UI to compare
make mlflow-ui
```

Filter by tag: `task=nli_binary_classification`

### Example 3: Hardware-Specific Training

```bash
# Train on different GPUs and compare
make nli-train-4090  # RTX 4090
make nli-train-3090  # RTX 3090

# Compare performance
make mlflow-ui
```

Filter by `gpu_name` tag to see hardware-specific results.

### Example 4: Reproduce a Run

From MLflow UI:

1. Select a run
2. View "Parameters" tab
3. Copy configuration
4. Create new config file or use CLI overrides

```bash
python src/training/train_nli_5fold.py \
    model.name=google/gemma-2-2b \
    model.pooling_strategy=mean \
    training.batch_size=8 \
    training.learning_rate=2e-5 \
    data.negative_ratio=1.0
```

---

## Model Registry

### What is Model Registry?

MLflow Model Registry is a centralized model store for:
- Versioning models
- Stage transitions (staging → production)
- Model lineage tracking
- Annotations and descriptions

### Registering Models

Models are **automatically registered** after 5-fold CV training:

```python
# Best model (highest F1) is registered as:
registered_model_name="gemma_nli_{experiment_name}"
```

### Viewing Registered Models

```bash
# List all registered models
make mlflow-models
```

Output:
```
Registered Models:
  gemma_nli_nli_binary_5fold
    v1: None
    v2: Production
  gemma_nli_nli_4090_optimized
    v1: Staging
```

### Promoting Models

In MLflow UI:

1. Go to "Models" tab
2. Select a model version
3. Click "Transition to: Production"

Or programmatically:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="gemma_nli_my_experiment",
    version=2,
    stage="Production"
)
```

### Loading Registered Models

```python
import mlflow

# Load latest version
model_uri = "models:/gemma_nli_nli_binary_5fold/latest"
model = mlflow.pytorch.load_model(model_uri)

# Load specific version
model_uri = "models:/gemma_nli_nli_binary_5fold/2"
model = mlflow.pytorch.load_model(model_uri)

# Load production model
model_uri = "models:/gemma_nli_nli_binary_5fold/Production"
model = mlflow.pytorch.load_model(model_uri)
```

---

## Comparing Experiments

### Using MLflow UI

1. **Launch UI**: `make mlflow-ui`
2. **Select runs**: Check boxes next to runs
3. **Click "Compare"**
4. **View comparison**:
   - Parameters diff
   - Metrics charts
   - Parallel coordinates plot

### Using MLflow API

```python
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Get all runs from experiment
experiment_id = "0"
runs = client.search_runs(experiment_id)

# Extract metrics
data = []
for run in runs:
    data.append({
        'run_id': run.info.run_id,
        'run_name': run.info.run_name,
        'mean_f1': run.data.metrics.get('cv_aggregate/mean_f1'),
        'mean_auc': run.data.metrics.get('cv_aggregate/mean_auc'),
        'batch_size': run.data.params.get('training.batch_size'),
        'learning_rate': run.data.params.get('training.learning_rate'),
    })

df = pd.DataFrame(data)
print(df.sort_values('mean_f1', ascending=False))
```

### Querying Best Runs

```bash
# Show recent runs sorted by F1
make mlflow-runs
```

Or programmatically:

```python
# Find best run by F1
best_run = client.search_runs(
    experiment_ids=["0"],
    order_by=["metrics.cv_aggregate/mean_f1 DESC"],
    max_results=1
)[0]

print(f"Best run: {best_run.info.run_name}")
print(f"F1: {best_run.data.metrics['cv_aggregate/mean_f1']:.4f}")
```

---

## Best Practices

### 1. Use Descriptive Run Names

```bash
python src/training/train_nli_5fold.py \
    output.experiment_name=gemma2b_mean_pooling_balanced_lr2e5
```

### 2. Tag Experiments Appropriately

Tags are automatically added, but you can add custom tags:

```python
tracker.set_tags({
    'researcher': 'john_doe',
    'purpose': 'hyperparameter_tuning',
    'priority': 'high',
})
```

### 3. Log Additional Context

Log important notes or observations:

```python
tracker.log_params({
    'notes': 'Testing impact of criterion length',
    'hypothesis': 'Full criteria will improve F1',
})
```

### 4. Clean Up Old Runs

```bash
# BE CAREFUL! This deletes ALL tracking data
make mlflow-clean
```

Or selectively delete from UI:
1. Select runs to delete
2. Click "Delete"

### 5. Backup Tracking Data

```bash
# Backup mlruns directory
tar -czf mlruns_backup_$(date +%Y%m%d).tar.gz mlruns/

# Or use MLflow export
mlflow experiments export --experiment-id 0 --output-dir backup/
```

---

## Troubleshooting

### MLflow UI Not Starting

**Error**: `No module named 'mlflow'`

**Solution**:
```bash
pip install mlflow>=2.8.0
```

**Error**: `Address already in use`

**Solution**: Use custom port
```bash
make mlflow-ui-custom PORT=5001
```

### Metrics Not Logging

**Issue**: Metrics appear as `N/A` in UI

**Causes**:
1. Tracker not initialized
2. Exception during logging
3. MLflow disabled in config

**Debug**:
```bash
# Check if MLflow is enabled
grep "MLflow tracking enabled" outputs/*/logs.txt

# Check for errors
grep "Failed to" outputs/*/logs.txt
```

**Solution**:
```yaml
# Ensure MLflow is enabled
mlflow:
  enabled: true
```

### Model Registry Not Working

**Error**: `Model registry not available`

**Cause**: Using file-based backend (default)

**Solution**: Model registry works with file backend, but may have limitations. For production, use database backend:

```yaml
mlflow:
  tracking_uri: postgresql://user:pass@localhost/mlflow
```

### Artifacts Not Appearing

**Issue**: Artifacts tab is empty

**Causes**:
1. Files not logged
2. Permission issues
3. Path problems

**Debug**:
```python
# Check if artifacts were logged
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
run = client.get_run("RUN_ID")
artifacts = client.list_artifacts(run.info.run_id)
print(artifacts)
```

### Slow UI Performance

**Issue**: UI is slow with many runs

**Solutions**:
1. Archive old experiments
2. Use database backend instead of file storage
3. Filter runs in UI
4. Clean up old runs: `make mlflow-clean`

---

## Advanced Features

### Remote Tracking Server

Set up a shared MLflow server for team collaboration:

```bash
# Start server
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Update config:
```yaml
mlflow:
  tracking_uri: http://mlflow-server:5000
```

### Automated Experiment Comparison

Create a script to automatically find best hyperparameters:

```python
import mlflow
import pandas as pd

def find_best_config(metric='cv_aggregate/mean_f1'):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["0"],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    best_run = runs[0]
    print(f"Best {metric}: {best_run.data.metrics[metric]:.4f}")
    print("\nBest Configuration:")
    for key, value in best_run.data.params.items():
        print(f"  {key}: {value}")

    return best_run

best = find_best_config()
```

### Integration with Weights & Biases

The `ExperimentTracker` class supports both MLflow and W&B:

```python
tracker = ExperimentTracker(
    experiment_name='gemma_nli',
    use_mlflow=True,
    use_wandb=True,  # Enable W&B
    wandb_project='llm-criteria-gemma',
)
```

---

## Summary

**Key Commands:**

```bash
# Training with MLflow
make nli-train              # 5-fold CV (MLflow enabled by default)
python ... --mlflow         # Simple training (requires flag)

# Viewing Results
make mlflow-ui              # Launch UI
make mlflow-list            # List experiments
make mlflow-runs            # Show recent runs
make mlflow-models          # List registered models

# Management
make mlflow-clean           # Delete all runs (careful!)
```

**What Gets Tracked:**
- ✅ All hyperparameters
- ✅ Hardware specs
- ✅ Dataset metadata
- ✅ Per-epoch metrics
- ✅ Per-fold results
- ✅ Aggregate CV metrics
- ✅ Model checkpoints
- ✅ Configuration files
- ✅ Training history

**Model Registry:**
- ✅ Automatic registration of best models
- ✅ Version management
- ✅ Stage transitions (staging/production)
- ✅ Model lineage tracking

For more information, see the [MLflow documentation](https://mlflow.org/docs/latest/index.html).
