# Makefile Commands Reference

Complete reference for all Makefile commands in the Gemma Encoder project.

## Quick Start

```bash
make ref         # Show quick reference card
make info        # Show project information
make help        # Show all available commands
```

---

## NLI Binary Classification Commands

### Core Training

| Command | Description | Time Estimate |
|---------|-------------|---------------|
| `make nli-test` | Test NLI dataset creation and text-pair formatting | 1-2 min |
| `make nli-quick` | Quick NLI test (2 folds, 3 epochs) | 15-30 min |
| `make nli-train` | Full NLI 5-fold CV training (production) | 2-4 hours |
| `make nli-simple` | Simple NLI training (single split, no CV) | 30-60 min |
| `make nli-quickstart` | Complete NLI workflow (test + train + results) | 20-40 min |

### Model Variants

| Command | Description |
|---------|-------------|
| `make nli-gemma-2b` | NLI 5-fold CV with Gemma-2B (faster, less memory) |
| `make nli-gemma-9b` | NLI 5-fold CV with Gemma-9B (better performance) |
| `make nli-unfreeze` | NLI with unfrozen encoder (full fine-tuning) |

### Data Variants

| Command | Description |
|---------|-------------|
| `make nli-imbalanced` | NLI with imbalanced data (3:1 negative:positive ratio) |
| `make nli-short-criteria` | NLI with short criterion descriptions (memory efficient) |
| `make nli-full-criteria` | NLI with full criterion descriptions (recommended) |

### Pooling Variants

| Command | Description |
|---------|-------------|
| `make nli-pooling-mean` | NLI with mean pooling (recommended) |
| `make nli-pooling-attention` | NLI with learnable attention pooling |

### Ablation Studies

| Command | Description | Output |
|---------|-------------|--------|
| `make nli-ablation-pooling` | Test all pooling strategies (mean, cls, max, attention) | `outputs/nli_ablation_*/` |
| `make nli-ablation-negatives` | Test negative ratios (0.5, 1.0, 2.0, 3.0) | `outputs/nli_neg_*/` |

### Results & Documentation

| Command | Description |
|---------|-------------|
| `make nli-show-results` | Show aggregate results from latest NLI run |
| `make nli-docs` | Display NLI documentation (README_NLI.md) |
| `make nli-summary` | Display refactoring summary |

---

## Original Multi-Class Task Commands

### Training

| Command | Description | Time Estimate |
|---------|-------------|---------------|
| `make train` | Train with original script (single split) | 1-2 hours |
| `make train-quick` | Quick test (2 folds, 3 epochs) | 20-40 min |
| `make train-5fold` | Full 5-fold CV (default: MentaLLaMA) | 3-5 hours |
| `make train-5fold-mentallama` | 5-fold CV with MentaLLaMA-chat-7B | 3-5 hours |
| `make train-5fold-gemma` | 5-fold CV with Gemma-2-9B | 4-6 hours |
| `make train-5fold-both` | Train both MentaLLaMA and Gemma sequentially | 7-11 hours |

### Experiments

| Command | Description |
|---------|-------------|
| `make train-gemma9b` | Train with Gemma-9B model |
| `make train-attention` | Train with attention pooling |
| `make train-10fold` | Train with 10-fold CV |
| `make exp-pooling-comparison` | Compare all pooling strategies |
| `make exp-learning-rates` | Test different learning rates |

---

## Setup & Installation

| Command | Description |
|---------|-------------|
| `make install` | Install project dependencies |
| `make install-dev` | Install with development dependencies (pytest, black, etc.) |

---

## Evaluation

| Command | Description | Usage |
|---------|-------------|-------|
| `make evaluate` | Evaluate trained model | `make evaluate CHECKPOINT=path/to/model.pt` |
| `make evaluate-best` | Evaluate best model from latest 5-fold run | `make evaluate-best` or `make evaluate-best RUN=outputs/run_dir/` |
| `make show-results` | Show aggregate 5-fold results | `make show-results` or `make show-results RUN=outputs/run_dir/` |

---

## Data Management

| Command | Description |
|---------|-------------|
| `make check-data` | Verify dataset files exist |
| `make data-stats` | Show dataset statistics |
| `make prepare-splits` | Create CV splits manually |

---

## Code Quality

| Command | Description |
|---------|-------------|
| `make lint` | Run linting checks with flake8 |
| `make format` | Format code with black |
| `make format-check` | Check code formatting without changes |
| `make type-check` | Run type checking with mypy |

---

## Testing

| Command | Description |
|---------|-------------|
| `make test` | Run all tests |
| `make test-models` | Test model imports |
| `make test-data` | Test data loading |
| `make test-imports` | Test all imports |
| `make nli-test` | Test NLI dataset creation |

---

## Cleanup

| Command | Description | Warning |
|---------|-------------|---------|
| `make clean` | Remove generated files (cache, .pyc, .log) | Safe |
| `make clean-outputs` | Remove training outputs | ⚠️ Deletes all training results |
| `make clean-all` | Remove all generated files and outputs | ⚠️ Destructive |

---

## System & GPU

| Command | Description |
|---------|-------------|
| `make check-gpu` | Check GPU availability and memory |
| `make check-env` | Check Python environment |

---

## Monitoring

| Command | Description |
|---------|-------------|
| `make tensorboard` | Launch TensorBoard (if logs exist) |
| `make watch-training` | Watch training logs in real-time |

---

## Documentation

| Command | Description |
|---------|-------------|
| `make docs` | Open documentation in browser |
| `make show-config` | Show current Hydra configuration |
| `make info` | Show project information |
| `make version` | Show version information |
| `make help` | Display all available commands |
| `make ref` | Show quick reference card |

---

## Quick Commands (Shortcuts)

| Command | Description |
|---------|-------------|
| `make quick-check` | Quick sanity check (check-data + test-imports + check-gpu) |
| `make full-pipeline` | Full training pipeline (install + check-data + train-5fold + show-results) |
| `make demo` | Quick demo (2 folds, 3 epochs) |

---

## Usage Examples

### Complete NLI Workflow
```bash
# Quick start
make nli-quickstart

# Manual steps
make install
make check-data
make check-gpu
make nli-test
make nli-quick
make nli-show-results
```

### Production NLI Training
```bash
make nli-train
make nli-show-results
```

### Experiment with Different Configurations
```bash
# Try different models
make nli-gemma-2b
make nli-gemma-9b

# Try different data balances
make nli-imbalanced

# Compare pooling strategies
make nli-ablation-pooling
```

### Original Task Workflow
```bash
make install
make check-data
make train-quick
make show-results
```

### Full Production Run (Original Task)
```bash
make train-5fold-mentallama
make show-results RUN=outputs/mentallama_5fold/
```

---

## Custom Configuration Examples

All NLI commands support Hydra configuration overrides:

```bash
# Custom model and batch size
python src/training/train_nli_5fold.py \
    model.name=google/gemma-2-9b \
    training.batch_size=4 \
    output.experiment_name=custom_run

# Custom negative ratio and criteria length
python src/training/train_nli_5fold.py \
    data.negative_ratio=2.0 \
    data.use_short_criteria=false \
    output.experiment_name=neg2_full_criteria

# Unfreeze encoder with lower batch size
python src/training/train_nli_5fold.py \
    model.freeze_encoder=false \
    training.batch_size=2 \
    training.num_epochs=15 \
    output.experiment_name=unfrozen_15epochs
```

---

## Common Workflows

### 1. First Time Setup
```bash
make install
make check-data
make check-gpu
make nli-test
```

### 2. Quick Experiment
```bash
make nli-quick
make nli-show-results
```

### 3. Production Training
```bash
make nli-train
make nli-show-results
```

### 4. Ablation Study
```bash
make nli-ablation-pooling
make nli-ablation-negatives
```

### 5. Model Comparison
```bash
make nli-gemma-2b
make nli-gemma-9b
# Compare results in outputs/nli_gemma*_5fold/
```

---

## Output Locations

| Command Pattern | Output Directory |
|----------------|------------------|
| `make nli-*` | `outputs/nli_*/` |
| `make train-5fold-*` | `outputs/*/` |
| `make nli-ablation-*` | `outputs/nli_ablation_*/` or `outputs/nli_neg_*/` |

Each output directory contains:
- `config.yaml` - Saved configuration
- `cv_folds/` - Cross-validation splits
- `fold_N_best.pt` - Best model for each fold
- `fold_N_history.json` - Training history
- `aggregate_results.json` - Cross-validation summary

---

## Tips

1. **Always check GPU first**: `make check-gpu`
2. **Test before full run**: `make nli-quick` before `make nli-train`
3. **Use quick reference**: `make ref` for common commands
4. **Monitor results**: `make nli-show-results` to check latest run
5. **Clean between runs**: `make clean` to remove cache files
6. **Read documentation**: `make nli-docs` for detailed info

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'pandas'` | `make install` |
| `CUDA out of memory` | Reduce batch size or use smaller model |
| `Data files not found` | `make check-data` to verify |
| `Command not found` | Ensure you're in the project root directory |
| `Import errors` | `make test-imports` to diagnose |

---

## See Also

- **README.md** - Original task documentation
- **README_NLI.md** - NLI task documentation
- **REFACTORING_SUMMARY.md** - Complete refactoring details
- **conf/config_nli.yaml** - NLI configuration file
