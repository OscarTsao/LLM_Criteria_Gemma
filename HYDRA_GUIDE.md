# Hydra Configuration Guide

This project uses [Hydra](https://hydra.cc/) for configuration management and supports 5-fold cross-validation.

## Quick Start

### Basic 5-fold CV Training
```bash
python src/training/train_gemma_hydra.py
```

### Override Parameters
```bash
# Change model size
python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b

# Adjust batch size and learning rate
python src/training/train_gemma_hydra.py training.batch_size=8 training.learning_rate=3e-5

# Change number of folds
python src/training/train_gemma_hydra.py cv.num_folds=10

# Use different pooling strategy
python src/training/train_gemma_hydra.py model.pooling_strategy=attention
```

### Use Experiment Configs
```bash
# Quick test (2 folds, 3 epochs)
python src/training/train_gemma_hydra.py experiment=quick_test

# Full 5-fold training
python src/training/train_gemma_hydra.py experiment=full_5fold
```

## Configuration Structure

```
conf/
├── config.yaml              # Main configuration
└── experiment/
    ├── quick_test.yaml      # Quick test config
    └── full_5fold.yaml      # Full 5-fold config
```

## Key Configuration Sections

### Model Configuration
```yaml
model:
  name: google/gemma-2b              # Model size
  pooling_strategy: mean             # Pooling method
  freeze_encoder: false              # Freeze encoder weights
  hidden_dropout_prob: 0.1           # Dropout rate
  classifier_hidden_size: null       # Hidden layer size (null = linear)
```

### Training Configuration
```yaml
training:
  num_epochs: 10                     # Training epochs
  batch_size: 16                     # Batch size
  learning_rate: 2e-5                # Learning rate
  weight_decay: 0.01                 # Weight decay
  warmup_ratio: 0.1                  # Warmup ratio
  use_class_weights: true            # Use class weighting
```

### Cross-Validation Configuration
```yaml
cv:
  enabled: true                      # Enable CV
  num_folds: 5                       # Number of folds
  stratified: true                   # Stratify by symptom
  save_fold_results: true            # Save individual fold results
```

### Data Configuration
```yaml
data:
  data_dir: /path/to/redsm5          # Dataset directory
  max_length: 512                    # Max sequence length
  num_folds: 5                       # Number of CV folds
  random_seed: 42                    # Random seed
```

## Output Structure

After training, outputs are organized as:

```
outputs/
└── gemma_5fold/                    # Experiment name
    ├── fold_0/
    │   ├── best_model.pt           # Best model checkpoint
    │   └── history.json            # Training history
    ├── fold_1/
    │   └── ...
    ├── fold_2/
    │   └── ...
    ├── fold_3/
    │   └── ...
    ├── fold_4/
    │   └── ...
    ├── cv_results.csv              # Cross-validation results
    └── aggregate_results.json      # Aggregate statistics
```

## Cross-Validation Results

The `aggregate_results.json` contains:
```json
{
  "mean_f1": 0.7234,
  "std_f1": 0.0123,
  "min_f1": 0.7102,
  "max_f1": 0.7401,
  "fold_results": [
    {"fold": 0, "best_val_f1": 0.7234, ...},
    ...
  ]
}
```

## Advanced Usage

### Multiple Runs with Different Seeds
```bash
for seed in 42 123 456; do
    python src/training/train_gemma_hydra.py \
        data.random_seed=$seed \
        output.experiment_name=gemma_5fold_seed${seed}
done
```

### Grid Search with Hydra Multirun
```bash
python src/training/train_gemma_hydra.py -m \
    model.pooling_strategy=mean,cls,attention \
    training.learning_rate=1e-5,2e-5,3e-5
```

### Custom Experiment Config
Create `conf/experiment/my_experiment.yaml`:
```yaml
# @package _global_

model:
  name: google/gemma-2-9b
  pooling_strategy: attention

training:
  num_epochs: 15
  batch_size: 8
  learning_rate: 1e-5

output:
  experiment_name: my_custom_experiment
```

Then run:
```bash
python src/training/train_gemma_hydra.py experiment=my_experiment
```

## Tips

1. **Start with quick_test**: Test your setup with 2 folds and 3 epochs
2. **Monitor GPU memory**: Adjust batch_size if you get OOM errors
3. **Use mixed precision**: Set `device.mixed_precision=true` for faster training
4. **Save space**: Set `output.save_best_only=true` to only keep best checkpoints
5. **Reproducibility**: Always set `data.random_seed` for reproducible results

## Hydra Features Used

- ✅ Configuration composition
- ✅ Command-line overrides
- ✅ Experiment configs
- ✅ Automatic output directory management
- ✅ Config resolution and validation

## Comparison: Original vs Hydra

| Feature | Original | Hydra |
|---------|----------|-------|
| Config | Single YAML | Composable configs |
| CLI args | Manual parsing | Automatic overrides |
| CV | Single run | Built-in 5-fold |
| Experiments | Manual tracking | Named experiments |
| Outputs | Manual paths | Auto-organized |

## References

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
