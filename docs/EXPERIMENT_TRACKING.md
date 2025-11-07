# Experiment Tracking Guide

This project supports experiment tracking with MLflow and Weights & Biases (W&B).

## Overview

### Why Track Experiments?

- **Reproducibility**: Track all hyperparameters and configurations
- **Comparison**: Compare different model versions and configurations
- **Collaboration**: Share results with team members
- **Visualization**: View metrics, plots, and model artifacts
- **Model Registry**: Manage model versions and deployment

### Supported Systems

1. **MLflow** (Default)
   - Open-source
   - Self-hosted or cloud
   - Model registry
   - Local or remote storage

2. **Weights & Biases** (Optional)
   - Cloud-based
   - Collaborative features
   - Advanced visualizations
   - Model versioning

## Quick Start

### Install Dependencies

```bash
# MLflow only
pip install mlflow

# Both MLflow and W&B
pip install mlflow wandb

# Add to requirements.txt
echo "mlflow>=2.8.0" >> requirements.txt
echo "wandb>=0.16.0" >> requirements.txt  # optional
```

### Basic Usage

```python
from utils.experiment_tracking import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker(
    experiment_name="gemma_redsm5",
    run_name="experiment_001",
    use_mlflow=True,
    use_wandb=False,
)

# Log configuration
config = {
    'model_name': 'google/gemma-2-2b',
    'batch_size': 16,
    'learning_rate': 2e-5,
}
tracker.log_params(config)

# Log metrics during training
for epoch in range(num_epochs):
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_f1': val_f1,
    }
    tracker.log_metrics(metrics, step=epoch)

# Log model
tracker.log_model('best_model.pt', 'gemma_classifier')

# Finish tracking
tracker.finish()
```

## MLflow Setup

### Local Setup

1. **Start MLflow UI**:
   ```bash
   mlflow ui
   ```
   Access at http://localhost:5000

2. **Configure tracking URI**:
   ```python
   tracker = ExperimentTracker(
       experiment_name="my_experiment",
       tracking_uri="./mlruns",  # Local
   )
   ```

### Remote Server Setup

1. **Start MLflow server**:
   ```bash
   mlflow server \
       --backend-store-uri sqlite:///mlflow.db \
       --default-artifact-root ./mlruns \
       --host 0.0.0.0 \
       --port 5000
   ```

2. **Connect from client**:
   ```python
   tracker = ExperimentTracker(
       experiment_name="my_experiment",
       tracking_uri="http://mlflow-server:5000",
   )
   ```

### MLflow with S3 Storage

```bash
# Set environment variables
export MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Start server with S3 backend
mlflow server \
    --backend-store-uri postgresql://user:password@postgres:5432/mlflow \
    --default-artifact-root s3://my-mlflow-bucket/artifacts \
    --host 0.0.0.0
```

## Weights & Biases Setup

### Create Account

1. Sign up at https://wandb.ai
2. Get your API key from https://wandb.ai/authorize
3. Login:
   ```bash
   wandb login
   # Or set environment variable
   export WANDB_API_KEY=your_api_key
   ```

### Basic Usage

```python
tracker = ExperimentTracker(
    experiment_name="gemma_redsm5",
    use_mlflow=False,
    use_wandb=True,
    wandb_project="llm-criteria-gemma",
    wandb_entity="your_username",
)
```

### Advanced Features

```python
# Log custom charts
tracker.wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true,
        preds=y_pred,
        class_names=class_names
    )
})

# Log images
tracker.wandb.log({"examples": [wandb.Image(img) for img in images]})

# Log tables
table = wandb.Table(
    columns=["text", "prediction", "label"],
    data=[[text, pred, label] for text, pred, label in zip(texts, preds, labels)]
)
tracker.wandb.log({"predictions": table})
```

## Integration with Training

### Modify Training Script

```python
# In train_gemma_hydra.py
from utils.experiment_tracking import ExperimentTracker

def main(cfg: DictConfig):
    # Initialize tracker
    tracker = ExperimentTracker(
        experiment_name=cfg.output.experiment_name,
        run_name=run_name,
        use_mlflow=cfg.get('mlflow', {}).get('enabled', True),
        use_wandb=cfg.get('wandb', {}).get('enabled', False),
        wandb_project=cfg.get('wandb', {}).get('project'),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        # Training loop
        for fold_idx in range(num_folds):
            for epoch in range(num_epochs):
                # Train and evaluate
                train_loss = train_epoch(...)
                val_metrics = evaluate(...)

                # Log metrics
                tracker.log_metrics({
                    f'fold_{fold_idx}/train_loss': train_loss,
                    f'fold_{fold_idx}/val_f1': val_metrics['f1'],
                }, step=epoch)

            # Log best model for this fold
            tracker.log_model(
                f'fold_{fold_idx}/best_model.pt',
                f'fold_{fold_idx}_model'
            )

        # Log aggregate results
        tracker.log_dict(aggregate_results, 'cv_results.json')

    finally:
        tracker.finish()
```

### Configuration

Add to `conf/config.yaml`:

```yaml
# Experiment tracking
mlflow:
  enabled: true
  tracking_uri: ./mlruns

wandb:
  enabled: false
  project: llm-criteria-gemma
  entity: null
```

## Viewing Results

### MLflow UI

```bash
# Start UI
mlflow ui

# Custom port
mlflow ui --port 5001

# Custom backend
mlflow ui --backend-store-uri sqlite:///custom.db
```

Navigate to http://localhost:5000 to view:
- Experiments and runs
- Metrics comparison
- Parameters
- Artifacts (models, plots)

### W&B Dashboard

1. Go to https://wandb.ai
2. Select your project
3. View:
   - Metric plots
   - System metrics
   - Hyperparameter importance
   - Model comparisons

## Best Practices

### Naming Conventions

```python
# Consistent experiment names
experiment_name = "gemma_redsm5_5fold"

# Descriptive run names
run_name = f"{model_name}_{timestamp}_{git_commit[:7]}"

# Hierarchical metrics
metrics = {
    'train/loss': train_loss,
    'train/accuracy': train_acc,
    'val/loss': val_loss,
    'val/f1_macro': val_f1,
    'val/f1_per_class/depressed_mood': f1_scores[0],
}
```

### What to Track

**Always log**:
- Hyperparameters
- Training/validation metrics
- Best model checkpoints
- Final evaluation results

**Consider logging**:
- Configuration files
- Training curves
- Confusion matrices
- Sample predictions
- System metrics (GPU/CPU usage)
- Code version (git commit)

**Avoid logging**:
- Every checkpoint (only best)
- Raw datasets
- Intermediate outputs
- Large artifacts unnecessarily

### Organization

```
experiments/
└── gemma_redsm5/
    ├── baseline/
    │   ├── run_001/
    │   ├── run_002/
    │   └── run_003/
    ├── hyperparam_search/
    │   ├── lr_sweep/
    │   └── batch_size_sweep/
    └── final/
        └── production_model/
```

## Comparing Experiments

### MLflow

```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.val_f1 > 0.75",
    order_by=["metrics.val_f1 DESC"],
)

# Compare metrics
best_runs = runs.nsmallest(5, 'metrics.val_f1')
print(best_runs[['run_id', 'metrics.val_f1', 'params.learning_rate']])
```

### W&B

```python
import wandb

api = wandb.Api()
runs = api.runs("username/project-name")

# Get best run
best_run = min(runs, key=lambda run: run.summary.get("val_loss", float('inf')))
print(f"Best run: {best_run.name}")
print(f"Val F1: {best_run.summary['val_f1']}")
```

## Troubleshooting

### MLflow Issues

**Problem**: Tracking URI not found
```bash
# Solution: Set explicit URI
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Problem**: Permission denied
```bash
# Solution: Fix permissions
chmod -R 755 mlruns/
```

### W&B Issues

**Problem**: Authentication failed
```bash
# Solution: Re-login
wandb login --relogin
```

**Problem**: Slow uploads
```bash
# Solution: Disable code logging
export WANDB_DISABLE_CODE=true
```

## Advanced Topics

### Custom Metrics

```python
# Log custom calculated metrics
tracker.log_metrics({
    'learning_rate': optimizer.param_groups[0]['lr'],
    'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm),
    'epoch_time_minutes': epoch_time / 60,
})
```

### Hyperparameter Sweeps

```python
# W&B Sweep configuration
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'values': [1e-5, 2e-5, 5e-5]},
        'batch_size': {'values': [8, 16, 32]},
    },
    'metric': {'name': 'val_f1', 'goal': 'maximize'}
}

sweep_id = wandb.sweep(sweep_config, project="llm-criteria-gemma")
wandb.agent(sweep_id, function=train)
```

### Model Registry

```python
# MLflow Model Registry
mlflow.pytorch.log_model(
    model,
    "model",
    registered_model_name="GemmaClassifier"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="GemmaClassifier",
    version=1,
    stage="Production"
)
```

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [W&B Documentation](https://docs.wandb.ai/)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [W&B Integrations](https://docs.wandb.ai/guides/integrations)
