  # Binary NLI Criteria Matching Task

## Overview

This implementation converts the original multi-class ReDSM5 classification task into a **binary Natural Language Inference (NLI) style criteria matching task**.

### Task Transformation

**Original Task (Multi-class):**
- Input: Reddit post
- Output: One of 10 classes (9 DSM-5 symptoms + SPECIAL_CASE)
- Examples: 1,484 posts

**New Task (Binary NLI):**
- Input: `[CLS] post [SEP] criterion [SEP]`
- Output: `matched (1)` or `unmatched (0)`
- Examples: 1,484 posts × 9 criteria = **13,356 pairs**

## Motivation

Binary criteria matching offers several advantages:

1. **More natural formulation**: Explicitly models the matching between posts and specific clinical criteria
2. **Better interpretability**: Can see which specific criterion a post matches
3. **Data efficiency**: Creates more training examples from the same data
4. **Flexible inference**: Can evaluate against new criteria without retraining

## Dataset Structure

### DSM-5 Criteria (9 symptoms)

Each of the 9 DSM-5 depression symptoms has a clinical criterion description:

1. **DEPRESSED_MOOD**: Depressed mood most of the day, feelings of sadness, emptiness, hopelessness
2. **ANHEDONIA**: Diminished interest or pleasure in activities
3. **APPETITE_CHANGE**: Significant weight/appetite changes
4. **SLEEP_ISSUES**: Insomnia or hypersomnia
5. **PSYCHOMOTOR**: Psychomotor agitation or retardation
6. **FATIGUE**: Fatigue or loss of energy
7. **WORTHLESSNESS**: Feelings of worthlessness or excessive guilt
8. **COGNITIVE_ISSUES**: Diminished ability to think or concentrate
9. **SUICIDAL_THOUGHTS**: Recurrent thoughts of death or suicidal ideation

### Input Format

Each example consists of:
- **Post text**: Reddit post from ReDSM5 dataset
- **Criterion text**: DSM-5 symptom criterion description
- **Label**: 1 if post matches criterion, 0 otherwise

The tokenizer automatically formats this as: `[CLS] post [SEP] criterion [SEP]`

### Data Splits

Splitting is done **by post** (not by pairs) to avoid data leakage:

- Train: ~70% of posts → ~9,348 pairs
- Validation: ~15% of posts → ~2,004 pairs
- Test: ~15% of posts → ~2,004 pairs

Each split contains all 9 criteria for each post.

### Class Balance

The dataset is naturally imbalanced:
- **Unmatched (0)**: ~85-90% (most post-criterion pairs don't match)
- **Matched (1)**: ~10-15% (only specific criteria match each post)

Class weights are used during training to handle this imbalance.

## Implementation

### File Structure

```
src/
├── data/
│   ├── dsm5_criteria.py           # DSM-5 criterion descriptions
│   ├── redsm5_nli_dataset.py      # Binary NLI dataset loader
│   └── redsm5_dataset.py          # Original multi-class dataset
├── models/
│   ├── gemma_encoder.py           # Gemma encoder (supports binary)
│   └── poolers.py                 # Pooling strategies
└── training/
    ├── train_nli_binary.py        # Training script for binary NLI
    ├── evaluate_nli_binary.py     # Evaluation script
    └── train_gemma_hydra.py       # Original multi-class training
```

### Key Components

#### 1. DSM-5 Criteria (`src/data/dsm5_criteria.py`)

Defines clinical criterion descriptions for each symptom:

```python
from data.dsm5_criteria import get_criterion_text, get_all_criteria

# Get criterion for a specific symptom
criterion = get_criterion_text('DEPRESSED_MOOD')

# Get all criteria
all_criteria = get_all_criteria()  # Returns dict of 9 criteria
```

#### 2. NLI Dataset (`src/data/redsm5_nli_dataset.py`)

Creates post-criterion pairs with binary labels:

```python
from data.redsm5_nli_dataset import load_redsm5_nli

# Load NLI dataset
train_dataset, val_dataset, test_dataset = load_redsm5_nli(
    data_dir='data/redsm5',
    tokenizer=tokenizer,
    max_length=512,
    test_size=0.15,
    val_size=0.15,
    random_seed=42,
)

# Each example contains:
sample = train_dataset[0]
# - sample['input_ids']: Tokenized [post, criterion] pair
# - sample['attention_mask']: Attention mask
# - sample['labels']: Binary label (0 or 1)
# - sample['post_id']: Original post ID
# - sample['criterion_name']: Which criterion (e.g., 'DEPRESSED_MOOD')
```

#### 3. Binary Classifier

The existing `GemmaClassifier` supports binary classification:

```python
from models.gemma_encoder import GemmaClassifier

model = GemmaClassifier(
    num_classes=2,  # Binary classification
    model_name="google/gemma-3-4b-it",
    pooling_strategy="mean",
)
```

## Usage

### Training

Train a binary NLI criteria matching model:

```bash
# Basic training
python src/training/train_nli_binary.py

# With custom parameters
python src/training/train_nli_binary.py \
    model.name=google/gemma-3-4b-it \
    training.batch_size=8 \
    training.learning_rate=2e-5 \
    training.num_epochs=5

# With class weighting
python src/training/train_nli_binary.py \
    training.use_class_weights=true
```

The training script will:
1. Load the ReDSM5 dataset and create post-criterion pairs
2. Split data by posts (avoiding leakage)
3. Train a binary classifier with the Gemma encoder
4. Save best model based on validation F1 score
5. Report per-criterion performance

### Evaluation

Evaluate a trained model on the test set:

```bash
python src/training/evaluate_nli_binary.py \
    --checkpoint outputs/nli_binary-google_gemma-3-4b-it-20241117/best_model.pt \
    --data-dir data/redsm5 \
    --batch-size 16
```

This generates:
- Overall binary classification metrics (accuracy, precision, recall, F1, AUC)
- Per-criterion performance analysis
- Confusion matrix
- Visualization plots
- Classification report

### Quick Test

Test the NLI dataset loading:

```bash
python tests/test_nli_dataset.py
```

This verifies:
- DSM-5 criteria loading
- NLI dataset creation
- Proper tokenization
- Label distribution
- Per-criterion statistics

## Expected Performance

### Baseline Metrics

For a well-trained binary NLI model:

- **Overall Accuracy**: 85-90%
- **Binary F1 (matched class)**: 60-75%
- **AUC**: 0.85-0.92

### Per-Criterion Variation

Performance varies by criterion based on:
- Prevalence in dataset
- Linguistic clarity
- Symptom ambiguity

Expected F1 scores by criterion:
- High performers (F1 > 0.70): DEPRESSED_MOOD, WORTHLESSNESS, SUICIDAL_THOUGHTS
- Medium performers (F1 0.50-0.70): ANHEDONIA, SLEEP_ISSUES, FATIGUE, COGNITIVE_ISSUES
- Lower performers (F1 < 0.50): PSYCHOMOTOR, APPETITE_CHANGE

## Comparison: Binary NLI vs Multi-class

| Aspect | Multi-class | Binary NLI |
|--------|-------------|------------|
| **Input** | Post only | Post + Criterion |
| **Output** | 1 of 10 classes | Matched/Unmatched |
| **Examples** | 1,484 | 13,356 |
| **Metric** | Macro F1 | Binary F1 |
| **Interpretability** | Class label | Explicit matching |
| **Extensibility** | Requires retraining | Can add criteria |

### When to Use Each

**Multi-class:**
- Faster inference (single forward pass per post)
- Simpler deployment
- Fixed set of symptoms

**Binary NLI:**
- Better interpretability (see which criterion matched)
- More training data from same dataset
- Can evaluate new criteria without retraining
- Better for threshold-based decisions

## Configuration

The training uses Hydra configuration from `conf/config.yaml`. Key parameters:

```yaml
model:
  name: google/gemma-3-4b-it
  pooling_strategy: mean
  use_dora: true
  dora_rank: 16

training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 5
  use_class_weights: true
  early_stopping_patience: 3

data:
  max_length: 512
  test_size: 0.15
  val_size: 0.15
  random_seed: 42
```

## Citation

If you use this binary NLI implementation, please cite both the original ReDSM5 dataset and the Gemma Encoder paper:

```bibtex
@article{bao2025redsm5,
  title={ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author={Bao, Eliseo and Pérez, Anxo and Parapar, Javier},
  journal={arXiv preprint arXiv:2508.03399},
  year={2025}
}

@article{suganthan2025gemma,
  title={Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks},
  author={Suganthan, Paul and Moiseev, Fedor and others},
  journal={arXiv preprint arXiv:2503.02656},
  year={2025}
}
```

## Troubleshooting

### Class Imbalance

If you see very high accuracy but low F1:
- Enable class weighting: `training.use_class_weights=true`
- Monitor precision/recall individually
- Consider adjusting the classification threshold

### Memory Issues

For large models or limited GPU memory:
- Reduce batch size: `training.batch_size=4`
- Enable gradient checkpointing: `model.use_gradient_checkpointing=true`
- Use smaller model: `model.name=google/gemma-2-2b-it`

### Data Leakage

The dataset splits by **post**, not pairs:
- All 9 criteria for a post are in the same split
- This prevents the model from memorizing posts
- Ensures realistic evaluation

## Future Enhancements

Potential improvements:

1. **Multi-task Learning**: Train on both binary NLI and multi-class objectives
2. **Contrastive Learning**: Learn better representations for criterion matching
3. **Few-shot Adaptation**: Evaluate on new mental health criteria
4. **Explainability**: Highlight which post segments match criteria
5. **Threshold Optimization**: Find optimal decision thresholds per criterion
