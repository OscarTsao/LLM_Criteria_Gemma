# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Gemma Encoder for DSM-5 Binary NLI Criteria Matching**

This project implements binary Natural Language Inference (NLI) for matching Reddit posts with DSM-5 depression criteria using the Gemma encoder architecture (arXiv:2503.02656) on the ReDSM5 dataset (arXiv:2508.03399).

**Core Task**: Given a `(post, criterion)` pair, predict if the post matches that DSM-5 symptom criterion (binary classification: matched vs unmatched).

**Key Innovation**: Converts Gemma's causal (decoder) attention to bidirectional (encoder) attention + DoRA parameter-efficient fine-tuning.

## Quick Start Commands

### Training
```bash
# Default: 1B model on 24GB GPU
make train-nli

# 4B model on 32GB+ GPU
make train-nli-4b

# Quick test (2 folds, 3 epochs)
make train-quick

# Custom configuration
python src/training/train_nli_binary.py model.name=google/gemma-3-1b-it training.batch_size=2

# With Hydra overrides
python src/training/train_nli_binary.py model.pooling_strategy=attention training.learning_rate=3e-5
```

### Evaluation & Testing
```bash
# Evaluate latest model
make evaluate-nli

# Show results
make show-results-nli

# Test dataset and imports
make test
make test-nli-dataset
make test-imports
```

### Data & Environment
```bash
# Check dataset and GPU
make check-data
make check-gpu
make check-env

# Dataset statistics
make data-stats
```

### Code Quality
```bash
# Linting and formatting
make lint
make format
make format-check
make type-check
```

### Cleanup
```bash
# Clean caches only
make clean

# DANGEROUS: Delete all training outputs
make clean-outputs

# Clean everything
make clean-all
```

## Architecture Overview

### Three-Layer Design

1. **Encoder Layer** (`src/models/gemma_encoder.py`)
   - **Bidirectional Attention**: Patches Gemma's causal mask to allow full context flow
   - **DoRA Fine-tuning**: Weight-decomposed low-rank adaptation on attention projections (q_proj, k_proj, v_proj, o_proj)
   - **Pooling**: Configurable strategies (mean, cls, max, attention) to get sequence representation
   - **Gradient Checkpointing**: Trades compute for memory to fit on 24GB GPUs

2. **Dataset Layer** (`src/data/redsm5_nli_dataset.py`)
   - **NLI Transformation**: Converts 1,484 posts × 9 criteria → 13,356 binary pairs
   - **Input Format**: `[CLS] post [SEP] criterion [SEP]` → tokenized
   - **Class Imbalance**: ~12% matched, ~88% unmatched (handled via class weights)
   - **Split by Post**: Ensures no post appears in both train and test

3. **Training Layer** (`src/training/train_nli_binary.py`)
   - **Hydra Configuration**: All hyperparameters in `conf/config.yaml`
   - **MLflow Tracking**: Dual saving (local checkpoints + MLflow artifacts)
   - **Mixed Precision**: bfloat16 for memory efficiency (no GradScaler needed)
   - **Early Stopping**: Monitors macro-F1 on validation set

### Critical Implementation Details

#### Bidirectional Attention Conversion
The `_enable_bidirectional_attention()` method in `GemmaEncoder` removes the causal mask while preserving padding masks. This is **essential** for encoder tasks and is what differentiates this from standard Gemma usage.

#### DoRA vs LoRA
DoRA decomposes weights into magnitude + direction: `W' = m' * (V + B*A) / ||V + B*A||`
- Applied to all attention projections (q_proj, k_proj, v_proj, o_proj)
- ~92% trainable parameters (vs 100% full fine-tuning)
- Rank=16, alpha=32.0 by default

#### Config Compatibility
Handles both `GemmaConfig` (flat) and `Gemma3Config` (nested) structures via `_get_text_config()` helper.

#### MLflow Integration
- Tracking URI: `sqlite:///mlflow.db`
- Artifacts: `mlruns/`
- Dual saving: local `best_model.pt` + MLflow `artifact_path="checkpoints"`
- **Important**: Use dictionary format for input_example: `{'input_ids': np.array, 'attention_mask': np.array}`

## Configuration System (Hydra)

### Base Configuration
`conf/config.yaml` - All defaults are here

### Key Config Sections
```yaml
model:
  name: google/gemma-3-1b-it  # Model checkpoint
  pooling_strategy: mean       # mean|cls|max|attention
  use_gradient_checkpointing: true

training:
  num_epochs: 100
  batch_size: 4                # Safe for 24GB GPU with 1B model
  learning_rate: 2e-5
  early_stopping_patience: 20

data:
  data_dir: ${hydra:runtime.cwd}/data/redsm5
  max_length: 512
  test_size: 0.15              # Split by post, not pairs
  val_size: 0.15

mlflow:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  log_models: true
```

### Override Examples
```bash
# Single override
python src/training/train_nli_binary.py model.name=google/gemma-3-4b-it

# Multiple overrides
python src/training/train_nli_binary.py \
  model.name=google/gemma-3-4b-it \
  training.batch_size=2 \
  training.learning_rate=3e-5

# Use experiment config
python src/training/train_nli_binary.py experiment=quick_test
```

## GPU Memory Requirements

| Model | Parameters | GPU Memory | Batch Size | Notes |
|-------|-----------|------------|-----------|-------|
| gemma-3-1b-it | 1B | ~12-15GB | 4 | **Default** - RTX 3090/4090 |
| gemma-3-4b-it | 4B | ~22-28GB | 2 | V100/A100 32GB |
| gemma-3-12b-it | 12B | ~48GB+ | 1 | A100 80GB |

Memory optimizations enabled by default:
- bfloat16 mixed precision
- Gradient checkpointing
- DoRA instead of full fine-tuning

## Common Issues & Solutions

### Issue: "MLflow expects dictionary but got tuple"
**Fix**: Input example must be dict of numpy arrays:
```python
input_example = {
    'input_ids': batch['input_ids'][:1].cpu().numpy(),
    'attention_mask': batch['attention_mask'][:1].cpu().numpy()
}
```

### Issue: "Checkpoint version mismatch"
**Fix**: Logger auto-increment was broken. Ensure logger created before callbacks:
```python
# Correct order:
logger = TensorBoardLogger(save_dir="outputs", name=cfg.data.name)
checkpoint_callback = ModelCheckpoint(dirpath=f"outputs/{logger.version}/checkpoints")
```

### Issue: "CUDA out of memory"
**Solutions**:
1. Reduce batch size: `training.batch_size=2` or `=1`
2. Use smaller model: `model.name=google/gemma-3-1b-it`
3. Enable gradient checkpointing (already default)
4. Reduce sequence length: `data.max_length=256`

### Issue: "FutureWarning: torch.cuda.amp.autocast deprecated"
**Fix**: Use new API:
```python
from torch.amp import autocast
with autocast('cuda', dtype=torch.bfloat16):
    outputs = model(input_ids, attention_mask)
```

## Dataset Structure

```
data/redsm5/
├── redsm5_posts.csv          # 1,484 Reddit posts
├── redsm5_annotations.csv    # Symptom annotations
└── README.md

data/DSM5/
└── MDD_Criteria.json         # 9 DSM-5 symptom criteria descriptions
```

**Data Flow**:
1. Load posts + annotations
2. Create post-criterion pairs (1,484 × 9 = 13,356)
3. Split by post ID (not pairs) to avoid leakage
4. Tokenize: `[CLS] post [SEP] criterion [SEP]`
5. Apply class weights for imbalance

## Output Structure

```
outputs/
└── nli_binary-{model}-{timestamp}/
    ├── best_model.pt            # Best checkpoint (local)
    ├── training_results.json    # Metrics summary
    ├── config.yaml              # Full config used
    └── predictions/
        └── test_predictions.csv

mlruns/
└── {experiment_id}/
    └── {run_id}/
        ├── artifacts/
        │   ├── model/           # MLflow model artifact
        │   └── checkpoints/     # Checkpoint backups
        └── metrics/             # Training curves
```

## Testing

```bash
# All tests
pytest tests/ -v

# Specific test
python tests/test_nli_dataset.py

# Test imports only
make test-imports
```

**Test Coverage**:
- `test_nli_dataset.py`: Dataset creation, splitting, class balance
- Model imports and initialization
- Tokenization and data format

## Development Workflow

1. **Check environment**: `make quick-check`
2. **Quick test**: `make train-quick` (2 folds, 3 epochs)
3. **Full training**: `make train-nli`
4. **Monitor**: Watch `outputs/` or use MLflow UI: `mlflow ui`
5. **Evaluate**: `make evaluate-nli`
6. **Results**: `make show-results-nli`

## Model Architecture Flow

```
Input Text
    ↓
Tokenizer (Gemma tokenizer)
    ↓
Input IDs + Attention Mask
    ↓
Gemma Model (with bidirectional attention)
    ↓
Hidden States [batch, seq_len, hidden_size]
    ↓
Pooler (mean/cls/max/attention)
    ↓
Pooled Representation [batch, hidden_size]
    ↓
Dropout (0.1)
    ↓
Classifier Head (linear)
    ↓
Logits [batch, 2] → Softmax → Prediction
```

## Key Files to Understand

### Models
- `src/models/gemma_encoder.py` - Bidirectional attention + DoRA setup
- `src/models/dora.py` - DoRA layer implementation
- `src/models/poolers.py` - 4 pooling strategies

### Data
- `src/data/redsm5_nli_dataset.py` - Binary NLI dataset creation
- `src/data/dsm5_criteria.py` - DSM-5 criterion definitions
- `src/data/cv_splits.py` - Cross-validation utilities

### Training
- `src/training/train_nli_binary.py` - Main training loop with MLflow
- `src/training/evaluate_nli_binary.py` - Evaluation script

### Config
- `conf/config.yaml` - Hydra base configuration
- `conf/experiment/quick_test.yaml` - Example override

## Experiment Tracking (MLflow)

### Start UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### What's Logged
- **Parameters**: All Hydra config values
- **Metrics**: Loss, accuracy, precision, recall, F1 (macro/micro) per epoch
- **Artifacts**: Model checkpoints, config files, predictions
- **Tags**: Model name, task type

### Query Experiments
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["nli_binary_1b"])
```

## Performance Expectations

**Expected Results** (on ReDSM5 test set):
- Accuracy: 75-85%
- Macro F1: 0.70-0.80
- Per-criterion F1 varies by symptom prevalence

**Training Time** (approximate):
- 1B model, 24GB GPU: ~2-3 hours for 100 epochs
- 4B model, 32GB GPU: ~4-6 hours for 100 epochs
- Early stopping typically triggers around epoch 30-50

## Important Conventions

### Naming
- Model outputs: `nli_binary-{model_name}-{timestamp}`
- Experiment names: Match output.experiment_name in config
- Checkpoints: Always `best_model.pt` in run directory

### Splits
- **Always split by post**, not by pairs
- Test posts never seen during training
- Stratification by matched/unmatched ratio

### Mixed Precision
- Use bfloat16 (not float16)
- No GradScaler needed for bfloat16
- Better numerical stability than float16

### Class Weights
- Computed from training set label distribution
- Applied during loss calculation
- Critical for imbalanced dataset (88% unmatched)

## Documentation

- **Main README**: `README.md`
- **Binary NLI Details**: `docs/BINARY_NLI_CRITERIA_MATCHING.md`
- **Hydra Guide**: `docs/HYDRA_GUIDE.md`
- **Makefile Reference**: `docs/MAKEFILE_GUIDE.md`

## Citation

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
