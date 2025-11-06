# Running 5-Fold Cross-Validation

## Prerequisites

```bash
cd /media/cvrlab308/cvrlab308_4090/YuNing/LLM_Criteria_Gemma
pip install -r requirements.txt
```

## Quick Start - Run 5-Fold CV

```bash
python src/training/train_gemma_hydra.py
```

This will:
1. Create stratified 5-fold splits of ReDSM5 dataset
2. Train Gemma-2B encoder on each fold
3. Save best model for each fold
4. Compute aggregate statistics (mean F1, std, min, max)

## Expected Output

```
================================================================================
Gemma Encoder 5-Fold Cross-Validation
================================================================================

Device: cuda

Loading tokenizer: google/gemma-2b

Creating 5-fold cross-validation splits...
Fold 0: Train=1237, Val=310
Fold 1: Train=1237, Val=310
Fold 2: Train=1238, Val=309
Fold 3: Train=1238, Val=309
Fold 4: Train=1238, Val=309

================================================================================
Starting Fold 1/5
================================================================================
Loading google/gemma-2b...
Train samples: 1237
Val samples: 310

Epoch 1/10
Training: 100%|████████| 78/78 [02:15<00:00]
Evaluating: 100%|████████| 20/20 [00:15<00:00]
Train Loss: 1.2345
Val Loss: 0.9876
Val Accuracy: 0.7234
Val F1: 0.7102
✓ Best model saved (F1: 0.7102)

...

================================================================================
Cross-Validation Results
================================================================================
   fold  best_val_f1  final_train_loss  final_val_loss  final_val_accuracy
0     0       0.7234              0.234           0.456               0.745
1     1       0.7156              0.241           0.467               0.738
2     2       0.7301              0.229           0.443               0.752
3     3       0.7189              0.237           0.461               0.741
4     4       0.7267              0.232           0.449               0.748

Mean F1: 0.7229 ± 0.0056
Min F1: 0.7156
Max F1: 0.7301

Results saved to: outputs/gemma_5fold
================================================================================
```

## Output Files

```
outputs/gemma_5fold/
├── fold_0/
│   ├── best_model.pt          # Best checkpoint (F1: 0.7234)
│   └── history.json           # Training curves
├── fold_1/
│   └── ...
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── cv_results.csv             # Per-fold results
└── aggregate_results.json     # Summary statistics
```

## Load and Use Trained Models

```python
import torch
from src.models.gemma_encoder import GemmaClassifier

# Load best model from fold 0
checkpoint = torch.load('outputs/gemma_5fold/fold_0/best_model.pt')

model = GemmaClassifier(
    num_classes=10,
    model_name='google/gemma-2b',
    pooling_strategy='mean'
)
model.load_state_dict(checkpoint['model_state_dict'])

# Use for inference
model.eval()
# ...
```

## Customization

### Change Model Size
```bash
python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b
```

### Adjust Training
```bash
python src/training/train_gemma_hydra.py \
    training.num_epochs=15 \
    training.batch_size=8 \
    training.learning_rate=3e-5
```

### Different Pooling Strategy
```bash
python src/training/train_gemma_hydra.py model.pooling_strategy=attention
```

### More/Fewer Folds
```bash
python src/training/train_gemma_hydra.py cv.num_folds=10
```

### Quick Test (2 folds, 3 epochs)
```bash
python src/training/train_gemma_hydra.py experiment=quick_test
```

## Training Time Estimates

| Model | Folds | Epochs | GPU | Time |
|-------|-------|--------|-----|------|
| Gemma-2B | 5 | 10 | A100 | ~2 hours |
| Gemma-2B | 5 | 10 | RTX 4090 | ~3 hours |
| Gemma-9B | 5 | 10 | A100 | ~8 hours |

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
python src/training/train_gemma_hydra.py training.batch_size=8

# Or use gradient accumulation (TODO: add to config)
```

### Slow Training
```bash
# Enable mixed precision (default: enabled)
python src/training/train_gemma_hydra.py device.mixed_precision=true
```

### Different Random Seed
```bash
python src/training/train_gemma_hydra.py data.random_seed=123
```

## Next Steps

1. **Analyze results**: Check `cv_results.csv` and `aggregate_results.json`
2. **Compare pooling strategies**: Run with different `model.pooling_strategy`
3. **Ensemble predictions**: Combine predictions from all 5 folds
4. **Hyperparameter tuning**: Use Hydra multirun for grid search

See `HYDRA_GUIDE.md` for more advanced usage.
