# Makefile Quick Reference Card

## 🚀 Most Used Commands

```bash
make help              # Show all commands
make info              # Project information
make install           # Install dependencies
make train-5fold       # Run 5-fold CV (main command)
make train-quick       # Quick test (2 folds, 3 epochs)
make show-results      # Display CV results
```

## 📋 Command Categories

### Setup & Verification
| Command | Description | Time |
|---------|-------------|------|
| `make install` | Install all dependencies | ~2 min |
| `make check-gpu` | Check GPU availability | instant |
| `make check-data` | Verify dataset exists | instant |
| `make test-imports` | Test all imports work | instant |
| `make quick-check` | Complete sanity check | ~10 sec |

### Training
| Command | Description | Time |
|---------|-------------|------|
| `make train-quick` | 2 folds, 3 epochs | ~30 min |
| `make train-5fold` | 5 folds, 10 epochs | ~2-3 hrs |
| `make train-10fold` | 10 folds, 10 epochs | ~4-6 hrs |
| `make train-attention` | With attention pooling | ~2-3 hrs |

### Evaluation
| Command | Description |
|---------|-------------|
| `make evaluate-best` | Evaluate best model |
| `make show-results` | Show aggregate stats |
| `make evaluate CHECKPOINT=path` | Eval specific checkpoint |

### Data
| Command | Description |
|---------|-------------|
| `make check-data` | Verify files exist |
| `make data-stats` | Show statistics |
| `make prepare-splits` | Create CV splits |

### Code Quality
| Command | Description |
|---------|-------------|
| `make format` | Format with black |
| `make lint` | Run flake8 |
| `make test` | Run pytest |

### Cleanup
| Command | Description |
|---------|-------------|
| `make clean` | Remove temp files |
| `make clean-outputs` | ⚠️ Delete results |
| `make clean-all` | ⚠️ Delete everything |

## 🎯 Common Workflows

### First Time Setup
```bash
make install
make check-data
make check-gpu
make test-imports
```

### Quick Test Run
```bash
make train-quick        # ~30 minutes
make show-results
```

### Full Training
```bash
make train-5fold        # ~2-3 hours
make show-results
```

### Development
```bash
make format
make lint
make test
```

## 📊 Expected Outputs

### After `make train-5fold`
```
outputs/gemma3_it_5fold/
├── fold_0/best_model.pt
├── fold_1/best_model.pt
├── fold_2/best_model.pt
├── fold_3/best_model.pt
├── fold_4/best_model.pt
├── cv_results.csv
└── aggregate_results.json
```

### `make show-results`
```json
{
  "mean_f1": 0.7229,
  "std_f1": 0.0056,
  "min_f1": 0.7156,
  "max_f1": 0.7301
}
```

## 🔧 Advanced Usage

### Custom Parameters (use Hydra directly)
```bash
python src/training/train_gemma_hydra.py \
    model.name=google/gemma-3-4b-it \
    training.batch_size=8 \
    training.learning_rate=3e-5 \
    cv.num_folds=10
```

### Evaluate Specific Model
```bash
make evaluate CHECKPOINT=outputs/gemma3_it_5fold/fold_0/best_model.pt
```

### Run Experiments
```bash
make exp-pooling-comparison    # Compare pooling strategies
make exp-learning-rates        # Test different LRs
```

## ⚡ Quick Tips

1. **Always check first**: `make quick-check`
2. **Start small**: `make train-quick` before full training
3. **Monitor GPU**: `make check-gpu` during training
4. **Save results**: Don't run `make clean-outputs`!
5. **Use help**: `make help` shows all commands

## 🐛 Troubleshooting

### Command not found
```bash
sudo apt-get install build-essential  # Ubuntu
brew install make                      # macOS
```

### Import errors
```bash
make test-imports    # Check what's wrong
make install         # Reinstall
```

### GPU issues
```bash
make check-gpu       # Check availability
```

### Out of memory
Use smaller batch size with Hydra:
```bash
python src/training/train_gemma_hydra.py training.batch_size=8
```

## 📖 More Info

- `make help` - Full command list
- `MAKEFILE_GUIDE.md` - Detailed guide
- `HYDRA_GUIDE.md` - Configuration guide
- `RUN_5FOLD.md` - Training guide

---
**Print this page for quick reference!**
