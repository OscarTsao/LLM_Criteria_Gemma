# NLI-Style Binary Criteria Matching with Gemma Encoder

This document describes the **NLI-style binary classification** task for DSM-5 criteria matching, refactored from the original multi-class symptom classification.

## Overview

**Task**: Binary classification of (post, criterion) pairs

**Input Format**: `[CLS] post [SEP] criterion [SEP]` (NSP/NLI-style text pairs)

**Output**: Binary label
- `0` = **unmatched**: Post does NOT exhibit the criterion
- `1` = **matched**: Post DOES exhibit the criterion

**Dataset**: ReDSM5 transformed into NLI pairs with balanced positive/negative samples

## Key Differences from Original Task

| Aspect | Original Task | NLI Task |
|--------|--------------|----------|
| **Input** | Single text (post only) | Text pair (post + criterion) |
| **Format** | `[CLS] post [SEP]` | `[CLS] post [SEP] criterion [SEP]` |
| **Labels** | 10 classes (symptom types) | 2 classes (matched/unmatched) |
| **Task Type** | Multi-class classification | Binary NLI/matching |
| **# Samples** | ~1,500 posts | ~3,000+ pairs (with negatives) |
| **Evaluation** | Per-symptom F1 | Binary F1, AUC-ROC |

## Architecture

The model architecture remains the same as the paper:
- **Encoder**: Gemma decoder with bidirectional attention
- **Pooling**: Mean pooling (or others)
- **Classifier**: 2-class head (binary: unmatched/matched)

```python
from src.models.gemma_encoder import GemmaClassifier

model = GemmaClassifier(
    num_classes=2,  # Binary classification
    model_name="google/gemma-2-2b",
    pooling_strategy="mean",
    freeze_encoder=True,
)
```

## Dataset Creation

### Positive Pairs (Matched)
For each annotation `(post_id, symptom_label)`:
- Pair the post with its correct criterion description
- Label: `1` (matched)

Example:
```
Post: "I feel so tired all the time, no energy to do anything..."
Criterion: "The patient reports fatigue or loss of energy nearly every day..."
Label: 1 (matched)
```

### Negative Pairs (Unmatched)
For each post:
- Randomly sample a different (incorrect) criterion
- Label: `0` (unmatched)

Example:
```
Post: "I feel so tired all the time, no energy to do anything..."
Criterion: "The patient has recurrent thoughts of death or suicidal ideation..."
Label: 0 (unmatched)
```

### Balancing
- `negative_ratio=1.0`: Equal positive and negative samples (balanced)
- `negative_ratio=2.0`: 2x more negatives (realistic imbalance)
- `negative_ratio=3.0`: 3x more negatives (hard negatives)

## Quick Start

### 1. Test Dataset Creation

```bash
python scripts/test_nli_dataset.py
```

This will:
- Verify criterion text loading
- Create NLI pairs from ReDSM5 data
- Show examples of positive/negative pairs
- Test tokenization with text-pair format

### 2. Simple Training (Single Split)

```bash
python src/training/train_nli_simple.py \
  --data_dir data/redsm5 \
  --model_name google/gemma-2-2b \
  --batch_size 8 \
  --epochs 10 \
  --output_dir outputs/nli_simple \
  --freeze_encoder
```

### 3. Full 5-Fold Cross-Validation

```bash
# Quick test (2 folds, 3 epochs)
python src/training/train_nli_5fold.py experiment=nli_quick_test

# Full 5-fold CV
python src/training/train_nli_5fold.py experiment=nli_full_5fold

# Imbalanced dataset (3:1 negative:positive)
python src/training/train_nli_5fold.py experiment=nli_imbalanced
```

### 4. Custom Configuration

```bash
python src/training/train_nli_5fold.py \
  model.name=google/gemma-2-9b \
  training.batch_size=4 \
  data.negative_ratio=2.0 \
  data.use_short_criteria=false
```

## Configuration

Edit `conf/config_nli.yaml` or use command-line overrides:

```yaml
model:
  name: google/gemma-2-2b
  pooling_strategy: mean
  freeze_encoder: true

training:
  num_epochs: 20
  batch_size: 8
  learning_rate: 2e-5
  use_class_weights: true  # Important for balance

data:
  negative_ratio: 1.0  # 1.0 = balanced
  use_short_criteria: false  # false = full descriptions

cv:
  num_folds: 5
```

## Hardware Optimization

The repository includes automatic hardware detection and optimization for different GPU configurations:

### Quick Start

```bash
# Check your hardware and get recommendations
make check-hardware

# Automatic hardware detection
make nli-train-auto

# Or use a specific GPU profile:
make nli-train-4090      # RTX 4090 (24GB)
make nli-train-3090      # RTX 3090 (24GB)
make nli-train-low-mem   # Low-memory GPUs (8-12GB)
```

### Hardware-Specific Configurations

**RTX 4090 / A6000 (24GB VRAM)**
- Batch size: 16
- Gradient checkpointing: Disabled
- Compile: Enabled (torch.compile)
- Expected: ~15-20 samples/sec

**RTX 3090 (24GB VRAM)**
- Batch size: 12
- Gradient checkpointing: Disabled
- Expected: ~12-18 samples/sec

**Low-Memory GPUs (8-12GB VRAM)**
- Batch size: 4
- Gradient checkpointing: Enabled
- Gradient accumulation: 4 steps (effective batch size: 16)
- Expected: ~6-10 samples/sec

### Automatic Optimizations

The hardware optimizer automatically applies:
- ✅ **Mixed precision (bfloat16)** - 2x faster training
- ✅ **TF32 on Ampere GPUs** - Up to 8x faster matmul
- ✅ **cuDNN benchmark mode** - Auto-tunes algorithms
- ✅ **Optimal DataLoader workers** - Based on GPU memory
- ✅ **Model compilation** - 10-30% speedup (PyTorch 2.0+)

### Manual Override

```bash
python src/training/train_nli_5fold.py \
    hardware=gpu_4090 \
    training.batch_size=20 \
    optimization.compile=true
```

**See [HARDWARE_OPTIMIZATION.md](HARDWARE_OPTIMIZATION.md) for complete guide.**

## DSM-5 Criteria Descriptions

The repository includes two versions of DSM-5 criterion texts:

### Full Descriptions (`use_short_criteria=false`)
```python
DEPRESSED_MOOD: "The patient exhibits a depressed mood most of the day,
nearly every day, as indicated by subjective report or observation by others.
This includes feeling sad, empty, or hopeless."
```

### Short Descriptions (`use_short_criteria=true`)
```python
DEPRESSED_MOOD: "Depressed mood most of the day, nearly every day."
```

**Recommendation**: Use full descriptions for better semantic matching, unless constrained by sequence length.

## Expected Results

Based on similar NLI tasks and the Gemma Encoder paper:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| **Accuracy** | 75-85% | Depends on negative ratio |
| **Precision** | 70-80% | Higher with class weights |
| **Recall** | 70-80% | Balanced dataset helps |
| **F1** | 70-80% | Primary metric |
| **AUC-ROC** | 0.80-0.90 | Discrimination ability |

**Factors affecting performance**:
- `negative_ratio`: Higher ratio (more negatives) is harder
- `use_short_criteria`: Full criteria provide more signal
- `freeze_encoder`: Frozen encoder faster but may limit performance
- Model size: Gemma-9B > Gemma-2B

## Output Structure

After training, outputs are organized as:

```
outputs/nli_full_5fold/
├── config.yaml                    # Saved configuration
├── cv_folds/                      # Cross-validation splits
│   ├── nli_pairs_full.csv         # All NLI pairs
│   ├── fold_1_train.csv           # Fold 1 training data
│   ├── fold_1_val.csv             # Fold 1 validation data
│   ├── fold_2_train.csv
│   ├── fold_2_val.csv
│   ├── ...
│   └── nli_cv_folds_metadata.json # Fold statistics
├── fold_1_best.pt                 # Best model for fold 1
├── fold_1_history.json            # Training history fold 1
├── fold_2_best.pt
├── fold_2_history.json
├── ...
└── aggregate_results.json         # Cross-validation summary
```

### Aggregate Results Example

```json
{
  "fold_results": [
    {"fold": 1, "best_val_f1": 0.7654, "best_val_auc": 0.8421},
    {"fold": 2, "best_val_f1": 0.7801, "best_val_auc": 0.8567},
    ...
  ],
  "mean_f1": 0.7728,
  "std_f1": 0.0123,
  "mean_auc": 0.8494,
  "std_auc": 0.0089
}
```

## File Structure

```
LLM_Criteria_Gemma/
├── src/
│   ├── data/
│   │   ├── dsm5_criteria.py           # NEW: Criterion descriptions
│   │   ├── redsm5_nli_dataset.py      # NEW: NLI dataset & pair creation
│   │   └── nli_cv_splits.py           # NEW: 5-fold CV for NLI
│   ├── training/
│   │   ├── train_nli_simple.py        # NEW: Simple training script
│   │   └── train_nli_5fold.py         # NEW: 5-fold CV training
│   └── models/
│       ├── gemma_encoder.py           # Unchanged (supports binary)
│       └── poolers.py                 # Unchanged
├── conf/
│   ├── config_nli.yaml                # NEW: NLI task config
│   └── experiment/
│       ├── nli_quick_test.yaml        # NEW: Quick test config
│       ├── nli_full_5fold.yaml        # NEW: Full 5-fold config
│       └── nli_imbalanced.yaml        # NEW: Imbalanced data config
├── scripts/
│   └── test_nli_dataset.py            # NEW: Dataset verification
└── README_NLI.md                      # This file
```

## Comparison with Paper

| Component | Paper | This Implementation |
|-----------|-------|---------------------|
| **Architecture** | Gemma Encoder (bidirectional) | ✓ Implemented |
| **Pooling** | Mean, First-K, Last-K, Attention | ✓ Implemented |
| **Tasks Tested** | GLUE (SST-2, MNLI, QQP, etc.) | **NLI-style criteria matching** |
| **Input Format** | Text pairs for NLI tasks | ✓ `[CLS] post [SEP] criterion [SEP]` |
| **Loss** | CrossEntropyLoss | ✓ With class weights |
| **Evaluation** | GLUE metrics | ✓ Binary F1, AUC |

**Alignment**: This NLI implementation **now matches the paper's methodology** for NLI-style tasks (like MNLI). The original multi-class task was a single-text classification task (like SST-2).

## Advanced Usage

### Custom Negative Sampling Strategy

Modify `create_nli_pairs()` in `src/data/redsm5_nli_dataset.py` to:
- Sample hard negatives (similar symptoms)
- Use all negatives (all 9 incorrect criteria per post)
- Apply domain-specific sampling

### Multi-Task Learning

Extend to jointly train on:
- Binary matching (matched/unmatched)
- Symptom classification (which symptom)
- Severity prediction (mild/moderate/severe)

### Ensemble Methods

Train multiple folds and ensemble predictions:
```python
# Average probabilities from all 5 folds
ensemble_probs = np.mean([fold1_probs, fold2_probs, ...], axis=0)
```

## Citation

If you use this NLI-style implementation, please cite:

```bibtex
@article{suganthan2025gemma,
  title={Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks},
  author={Suganthan, Paul and Moiseev, Fedor and others},
  journal={arXiv preprint arXiv:2503.02656},
  year={2025}
}

@article{bao2025redsm5,
  title={ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author={Bao, Eliseo and Pérez, Anxo and Parapar, Javier},
  journal={arXiv preprint arXiv:2508.03399},
  year={2025}
}
```

## Troubleshooting

### Issue: Class imbalance warnings
**Solution**: Ensure `use_class_weights=true` in config

### Issue: Out of memory
**Solution**:
- Reduce `batch_size`
- Use `freeze_encoder=true`
- Enable `use_gradient_checkpointing=true`
- Use shorter criteria (`use_short_criteria=true`)

### Issue: Poor performance on negatives
**Solution**:
- Increase `negative_ratio` during training
- Use hard negative sampling
- Ensure criterion descriptions are distinct

### Issue: Tokenizer doesn't support text pairs
**Solution**: The code uses `tokenizer(text1, text2, ...)` which automatically handles text-pair encoding for all HuggingFace tokenizers.

## License

Apache 2.0 (following ReDSM5 dataset license)
