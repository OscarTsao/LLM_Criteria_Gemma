# Gemma Encoder for DSM-5 Criteria Matching

Implementation of the Gemma Encoder architecture from the paper ["Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks"](https://arxiv.org/abs/2503.02656) applied to **binary NLI criteria matching** for DSM-5 mental health assessment using the ReDSM5 dataset.

## Overview

This project implements a **binary Natural Language Inference (NLI) system** for matching social media posts with DSM-5 depression criteria. Given a post-criterion pair, the model predicts whether the post matches that criterion.

**Task**: Binary classification for post-criterion pairs
- **Input**: `[CLS] post [SEP] criterion [SEP]`
- **Output**: `matched` (1) or `unmatched` (0)
- **Training examples**: 1,484 posts × 9 criteria = 13,356 pairs
- **Evaluation**: Per-criterion precision, recall, F1; overall macro/micro metrics

### Why Binary NLI?

Binary NLI criteria matching offers several advantages:
1. **Interpretability**: Clear yes/no decisions for each criterion
2. **Flexibility**: Can evaluate any post against any criterion combination
3. **Scalability**: Easily extends to new criteria without retraining
4. **Clinical Relevance**: Mirrors clinical diagnostic process

## Key Features

### 1. DoRA Parameter-Efficient Fine-Tuning
Weight-Decomposed Low-Rank Adaptation (DoRA) enables efficient fine-tuning by decomposing weights into magnitude and direction components.

- **Trainable Parameters**: ~92% of original model (vs 100% full fine-tuning)
- **Memory Efficient**: Fits 1B model on 24GB GPUs with batch_size=4
- **Architecture**: Low-rank matrices (rank=16) applied to attention projections (q_proj, k_proj, v_proj, o_proj)

### 2. Bidirectional Attention
Converting Gemma's causal (unidirectional) attention to bidirectional attention dramatically improves performance on encoding tasks.

```python
from src.models.gemma_encoder import GemmaClassifier

# Binary NLI model with DoRA
model = GemmaClassifier(
    num_classes=2,  # Binary: matched vs unmatched
    model_name="google/gemma-3-1b-it",  # 1B model works on 24GB GPUs
    pooling_strategy="mean",
    use_dora=True,  # DoRA for parameter-efficient fine-tuning
    dora_rank=16,
    dora_alpha=32.0
)
```

### 3. Multiple Pooling Strategies
Implemented poolers from the paper:
- **Mean Pooling**: Average over all tokens (recommended)
- **CLS Pooling**: Use first token representation
- **Max Pooling**: Max pooling over sequence
- **Attention Pooling**: Learnable attention-weighted pooling

### 4. ReDSM5 Dataset
The ReDSM5 dataset contains 1,484 Reddit posts annotated for 9 DSM-5 depression symptoms:
- DEPRESSED_MOOD
- ANHEDONIA
- APPETITE_CHANGE
- SLEEP_ISSUES
- PSYCHOMOTOR
- FATIGUE
- WORTHLESSNESS
- COGNITIVE_ISSUES
- SUICIDAL_THOUGHTS

## Installation

```bash
# Clone the repository
cd /path/to/LLM_Criteria_Gemma

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training

1. **Prepare Data**: Ensure ReDSM5 dataset is in `data/redsm5/`
   ```
   data/redsm5/
   ├── redsm5_posts.csv
   └── redsm5_annotations.csv
   ```

2. **Train Binary NLI Model**:
   ```bash
   # Using make (recommended)
   make train-nli

   # Or directly
   python src/training/train_nli_binary.py
   ```

3. **Train with custom model**:
   ```bash
   # Use 4B model (requires 32GB+ GPU)
   python src/training/train_nli_binary.py model.name=google/gemma-3-4b-it

   # Adjust batch size for smaller GPUs
   python src/training/train_nli_binary.py training.batch_size=2
   ```

### Evaluation

```bash
# Evaluate best model
make evaluate-nli

# Or specify checkpoint
python src/training/evaluate_nli_binary.py --checkpoint outputs/nli_binary-*/best_model.pt
```

### Testing

```bash
# Test dataset creation and loading
python tests/test_nli_dataset.py
```

## Project Structure

```
LLM_Criteria_Gemma/
├── data/
│   ├── DSM5/
│   │   └── MDD_Criteria.json     # DSM-5 criteria definitions
│   └── redsm5/
│       ├── redsm5_posts.csv
│       └── redsm5_annotations.csv
├── src/
│   ├── models/
│   │   ├── poolers.py            # Pooling strategies
│   │   ├── gemma_encoder.py      # Bidirectional Gemma encoder
│   │   └── dora.py               # DoRA parameter-efficient fine-tuning
│   ├── data/
│   │   ├── dsm5_criteria.py      # DSM-5 criterion descriptions
│   │   ├── redsm5_nli_dataset.py # Binary NLI dataset loader
│   │   ├── cv_splits.py          # Cross-validation utilities
│   │   └── annotations_utils.py  # Annotation helpers
│   ├── training/
│   │   ├── train_nli_binary.py   # Binary NLI training
│   │   └── evaluate_nli_binary.py # Binary NLI evaluation
│   └── utils/
│       └── logger.py             # Logging utilities
├── tests/
│   └── test_nli_dataset.py       # Test binary NLI dataset
├── docs/
│   └── BINARY_NLI_CRITERIA_MATCHING.md  # Detailed documentation
├── conf/
│   ├── config.yaml               # Hydra defaults
│   └── experiment/
│       └── quick_test.yaml       # Example overrides
├── outputs/                       # Training outputs
├── Makefile                      # Convenient make commands
├── requirements.txt
└── README.md
```

## Configuration

Edit `conf/config.yaml` (Hydra defaults) or add overrides under `conf/experiment/` to customize:
- Model checkpoint (`google/gemma-3-1b-it` by default for 24GB GPUs)
- DoRA settings (rank, alpha, dropout)
- Pooling strategy
- Hyperparameters (learning rate, dropout, batch size)
- Data splits and CV settings

## Model Options

The project targets the Gemma-3 family. Override `model.name` to explore other checkpoints.

| Model | Parameters | GPU Memory* | DoRA Layers | Notes |
|-------|-----------|-------------|-------------|-------|
| google/gemma-3-1b-it | 1B | ~12GB | 104 | **Default** - Works on 24GB GPUs with batch_size=4 |
| google/gemma-3-4b-it | 4B | ~22GB | 217 | Requires 32GB+ GPU or batch_size=1 |
| google/gemma-3-12b-it | 12B | ~48GB+ | ~650 | Requires 80GB+ GPU |

<sub>*Approximate memory when fine-tuning with bfloat16 mixed precision, DoRA enabled, and gradient checkpointing.</sub>

**GPU Requirements:**
- **24GB GPU (RTX 3090/4090)**: Use `google/gemma-3-1b-it` with batch_size=4
- **32GB+ GPU (V100/A100)**: Can use `google/gemma-3-4b-it` with batch_size=2-4
- **80GB+ GPU (A100-80GB)**: Can use `google/gemma-3-12b-it`

## Results

Expected performance on ReDSM5 binary NLI criteria matching:
- **Accuracy**: 75-85%
- **Macro F1**: 0.70-0.80
- **Per-criterion F1**: Varies by symptom prevalence

The binary NLI formulation provides:
- Interpretable per-criterion predictions
- Balanced class distribution (unlike multi-class)
- More training examples (13K vs 1.5K)
- Better generalization to unseen posts

## Key Implementation Details

### DoRA (Weight-Decomposed Low-Rank Adaptation)
From `src/models/dora.py`:
```python
class DoRALayer(nn.Module):
    """
    DoRA decomposes weight W into:
        W = m * V
    where m is magnitude and V is direction (normalized).

    The adapted weight becomes:
        W' = m' * (V + B*A) / ||V + B*A||
    where A and B are low-rank matrices.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.magnitude = nn.Parameter(torch.ones(out_features))
```

### Gemma3Config Compatibility
The implementation handles both flat (GemmaConfig) and nested (Gemma3Config) configuration structures:
```python
def _get_text_config(self):
    """Get text config, handling both Gemma and Gemma3 config structures."""
    if hasattr(self.model.config, 'text_config'):
        return self.model.config.text_config  # Gemma3Config - nested
    else:
        return self.model.config  # GemmaConfig - flat
```

### Bidirectional Attention
From `src/models/gemma_encoder.py`:
```python
def _enable_bidirectional_attention(self):
    """
    Critical modification: removes causal mask to allow
    bidirectional context flow during fine-tuning.
    """
    # Patches attention layers to disable causal masking
    # while preserving padding masks
```

### Pooling Strategies
From `src/models/poolers.py`:
```python
# Mean pooling (recommended)
pooler = MeanPooler()

# Attention pooling with learnable query
pooler = AttentionPooler(hidden_size)
```

### Mixed Precision Training (bfloat16)
The implementation uses bfloat16 for memory efficiency without GradScaler:
```python
# bfloat16 has better numerical stability than float16
with autocast(dtype=torch.bfloat16):
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)

# No GradScaler needed for bfloat16
loss.backward()
optimizer.step()
```

### Binary NLI Dataset Creation
From `src/data/redsm5_nli_dataset.py`:
```python
# Creates post-criterion pairs with binary labels
# Positive examples: post matches annotated criterion
# Negative examples: post paired with non-matching criteria
dataset = ReDSM5NLIDataset(
    annotations_path='data/redsm5/redsm5_annotations.csv',
    posts_path='data/redsm5/redsm5_posts.csv',
    tokenizer=tokenizer,
    max_length=512
)
# Returns: (input_ids, attention_mask, label, criterion_name)
```

## Make Commands

Common operations via Makefile:

```bash
# Training
make train-nli          # Train binary NLI model (default 1B model)
make train-nli-4b       # Train with 4B model (requires 32GB+ GPU)

# Evaluation
make evaluate-nli       # Evaluate best model
make show-results-nli   # Show training results

# Data
make check-data         # Verify dataset files exist
make data-stats         # Show dataset statistics

# Development
make test               # Run all tests
make clean              # Remove generated files
make help               # Show all available commands
```

## Documentation

For detailed documentation on the binary NLI approach:
- [Binary NLI Criteria Matching Guide](docs/BINARY_NLI_CRITERIA_MATCHING.md)

## Citation

If you use this implementation, please cite the Gemma Encoder paper and ReDSM5 dataset:

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

## License

Apache 2.0 (following ReDSM5 dataset license)

## Contact

For questions about the implementation or results, please open an issue or contact the project maintainers.
