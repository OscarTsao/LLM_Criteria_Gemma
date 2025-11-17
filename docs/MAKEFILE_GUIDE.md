# Makefile Commands Guide

This project includes a comprehensive Makefile for easy project management.

## Quick Reference

```bash
make help              # Show all available commands
make info              # Show project information
make quick-check       # Verify setup (data, imports, GPU)
```

## Setup Commands

```bash
make install           # Install dependencies
make install-dev       # Install with dev tools (pytest, black, etc.)
make check-env         # Check Python environment
make check-gpu         # Check GPU availability and memory
```

## Training Commands

### Basic Training
```bash
make train             # Original training script (single split)
make train-5fold       # 5-fold cross-validation (default)
make train-quick       # Quick test (2 folds, 3 epochs)
```

### Model Variants
```bash
make train-attention   # Use attention pooling
make train-10fold      # 10-fold cross-validation
```

### Custom Training
```bash
# You can still use Hydra overrides:
python src/training/train_gemma_hydra.py model.name=google/gemma-3-4b-it training.batch_size=8
```

## Evaluation Commands

```bash
# Evaluate specific checkpoint
make evaluate CHECKPOINT=outputs/gemma3_it_5fold/fold_0/best_model.pt

# Evaluate best model from 5-fold CV
make evaluate-best

# Show aggregate results
make show-results
```

## Data Commands

```bash
make check-data        # Verify dataset files exist
make data-stats        # Show dataset statistics
make prepare-splits    # Create CV splits manually
```

### Example Output
```bash
$ make data-stats
Total samples: 1547

Symptom distribution:
DEPRESSED_MOOD         328
WORTHLESSNESS          311
SUICIDAL_THOUGHTS      165
ANHEDONIA              124
FATIGUE                124
...
```

## Code Quality Commands

```bash
make lint              # Run flake8 linting
make format            # Format code with black
make format-check      # Check formatting without changes
make type-check        # Run mypy type checking
```

## Testing Commands

```bash
make test              # Run all tests
make test-models       # Test model imports
make test-data         # Test data loading
make test-imports      # Test all imports
```

## Cleanup Commands

```bash
make clean             # Remove __pycache__, *.pyc, etc.
make clean-outputs     # Remove training outputs (WARNING: deletes results!)
make clean-all         # Remove everything
```

## Experiment Commands

### Pooling Strategy Comparison
```bash
make exp-pooling-comparison
```
Trains models with all pooling strategies (mean, cls, max, attention) using quick_test config.

Output: `outputs/pooling_mean/`, `outputs/pooling_cls/`, etc.

### Learning Rate Experiments
```bash
make exp-learning-rates
```
Tests learning rates: 1e-5, 2e-5, 3e-5, 5e-5 with quick_test config.

Output: `outputs/lr_1e-5/`, `outputs/lr_2e-5/`, etc.

## Monitoring Commands

```bash
make tensorboard       # Launch TensorBoard
make watch-training    # Tail training logs
```

## Documentation Commands

```bash
make docs              # Open README.md
make show-config       # Show Hydra configuration
```

## Quick Workflows

### First Time Setup
```bash
make install
make check-data
make check-gpu
make train-quick       # Test everything works
```

### Full Experiment
```bash
make install
make quick-check
make train-5fold
make show-results
```

### Quick Demo
```bash
make demo              # Runs train-quick + show-results
```

### Development Workflow
```bash
make install-dev
make format            # Format code
make lint              # Check code quality
make test              # Run tests
```

## Common Use Cases

### 1. Initial Setup
```bash
cd /path/to/LLM_Criteria_Gemma
make install
make check-data
make test-imports
make check-gpu
```

### 2. Quick Test Run
```bash
make train-quick       # 2 folds, 3 epochs (~30 min)
make show-results
```

### 3. Full 5-Fold Training
```bash
make train-5fold       # 5 folds, 10 epochs (~2-3 hours)
make show-results
```

### 4. Evaluate Model
```bash
make evaluate CHECKPOINT=outputs/gemma3_it_5fold/fold_0/best_model.pt
```

### 5. Compare Pooling Strategies
```bash
make exp-pooling-comparison
# Compare results in outputs/pooling_*/aggregate_results.json
```

### 6. Clean and Restart
```bash
make clean             # Keep training outputs
make clean-all         # Remove everything
```

## Environment Variables

You can set environment variables before make commands:

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 make train-5fold

# Set number of workers
NUM_WORKERS=4 make train-5fold
```

## Tips

1. **Check GPU first**: Always run `make check-gpu` before training
2. **Start small**: Use `make train-quick` to test your setup
3. **Monitor memory**: Use `make check-gpu` during training to monitor GPU memory
4. **Save outputs**: Don't run `make clean-outputs` unless you're sure!
5. **Use experiments**: Pre-configured experiment configs are easier than manual overrides

## Troubleshooting

### "make: command not found"
Install make:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS (via Homebrew)
brew install make
```

### "CHECKPOINT not specified" error
```bash
# Wrong:
make evaluate

# Correct:
make evaluate CHECKPOINT=path/to/model.pt
```

### "No module named 'src'"
Make sure you're in the project root:
```bash
cd /path/to/LLM_Criteria_Gemma
make train-5fold
```

### Imports failing
```bash
make test-imports      # Check what's wrong
make install           # Reinstall dependencies
```

## Advanced Usage

### Chain Multiple Commands
```bash
make clean install train-quick show-results
```

### Custom Make Targets
Add your own targets to the Makefile:
```makefile
my-experiment: ## My custom experiment
	python src/training/train_gemma_hydra.py \
		model.pooling_strategy=attention \
		training.learning_rate=3e-5 \
		cv.num_folds=10
```

### Parallel Experiments
```bash
# Terminal 1
make train-5fold

# Terminal 2 (different experiment name)
python src/training/train_gemma_hydra.py \
    output.experiment_name=parallel_exp \
    model.pooling_strategy=attention
```

## Complete Command List

Run `make help` to see all available commands organized by category:

```bash
$ make help

Usage:
  make <target>

General
  help                 Display this help message

Setup
  install              Install dependencies
  install-dev          Install with development dependencies

Training
  train                Train with original script
  train-5fold          Train with 5-fold CV
  train-quick          Quick test (2 folds, 3 epochs)
  ...

Evaluation
  evaluate             Evaluate trained model
  show-results         Show aggregate results
  ...

[And many more...]
```

## See Also

- `README.md` - Project overview
- `HYDRA_GUIDE.md` - Hydra configuration details
- `RUN_5FOLD.md` - 5-fold CV instructions
