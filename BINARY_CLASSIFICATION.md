# Binary Post-Criterion Matching

## Overview

This document describes the **binary classification** variant of the DSM-5 criteria matching task. Instead of predicting which symptom a post expresses (multi-class), we predict whether a post matches a given criterion (binary).

## Task Transformation

### Before: Multi-Class Classification

**Input:** Reddit post text
```
"I can't sleep at night, I toss and turn for hours"
```

**Output:** Symptom class (1 of 10)
```
SLEEP_ISSUES (class 3)
```

### After: Binary Matching

**Input:** (Post text, Criterion description)
```
Post: "I can't sleep at night, I toss and turn for hours"
Criterion: "Insomnia or hypersomnia nearly every day"
```

**Output:** Binary match (0 or 1)
```
1 (Match!)
```

**Alternative pair:**
```
Post: "I can't sleep at night, I toss and turn for hours"
Criterion: "Feelings of worthlessness or excessive guilt"
Output: 0 (No match)
```

## Why Binary Matching?

### Advantages

1. **More Realistic Task:** In practice, clinicians match symptoms to specific DSM-5 criteria, not just identify symptom categories.

2. **Flexible Inference:** Can evaluate against any criterion, including new ones not seen during training.

3. **Richer Training Signal:** Creates more training examples from the same data (N posts × 10 criteria = 10N pairs).

4. **Better for Class Imbalance:** Each post contributes balanced positive/negative examples.

5. **Interpretable:** "Does this post match this criterion?" is more interpretable than "Which symptom?"

### Disadvantages

1. **Computational Cost:** 10× more inference cost (must evaluate all 10 criteria per post).

2. **Class Imbalance (with 'all' sampling):** 1:9 ratio of positive to negative examples.

3. **Potential Redundancy:** Some criteria pairs may be highly correlated.

## Architecture

### Model Input

The model takes a **sentence pair** as input:

```
[CLS] post_text [SEP] criterion_description [SEP]
```

This is standard practice in transformers for tasks like:
- Natural Language Inference (NLI)
- Semantic Textual Similarity (STS)
- Question Answering

### Model Output

Binary logits:
```
(batch, 2) → [logit_no_match, logit_match]
```

Prediction:
```python
probs = softmax(logits)
predicted_class = argmax(probs)  # 0 or 1
confidence = probs[1]  # Probability of match
```

## Criterion Descriptions

We use **natural language descriptions** of each DSM-5 criterion:

| Criterion | Description |
|-----------|-------------|
| **DEPRESSED_MOOD** | Depressed mood most of the day, nearly every day (e.g., feels sad, empty, hopeless) |
| **ANHEDONIA** | Markedly diminished interest or pleasure in all or almost all activities |
| **APPETITE_CHANGE** | Significant weight loss or gain, or decrease or increase in appetite |
| **SLEEP_ISSUES** | Insomnia or hypersomnia nearly every day |
| **PSYCHOMOTOR** | Psychomotor agitation or retardation nearly every day |
| **FATIGUE** | Fatigue or loss of energy nearly every day |
| **WORTHLESSNESS** | Feelings of worthlessness or excessive guilt nearly every day |
| **COGNITIVE_ISSUES** | Diminished ability to think or concentrate, or indecisiveness |
| **SUICIDAL_THOUGHTS** | Recurrent thoughts of death or suicidal ideation |
| **SPECIAL_CASE** | Special case or expert discrimination required |

These descriptions are from DSM-5 diagnostic criteria for Major Depressive Disorder.

## Data Sampling Strategies

### 1. "All" Sampling (Default)

**Strategy:** Pair each post with **all 10 criteria**.

**Example:**
- Post: "I can't sleep"
- Pairs created: 10 (1 positive, 9 negative)
  - ("I can't sleep", SLEEP_ISSUES) → 1 ✓
  - ("I can't sleep", ANHEDONIA) → 0
  - ("I can't sleep", FATIGUE) → 0
  - ... (7 more negatives)

**Class Balance:** 1:9 (10% positive)

**Total Samples:** N_posts × 10

**Pros:**
- Complete coverage
- Consistent evaluation (all criteria tested)
- Simple

**Cons:**
- Severe class imbalance
- Many "easy" negatives

### 2. "Random" Sampling

**Strategy:** Pair each post with 1 positive + K random negatives.

**Example (K=3):**
- Post: "I can't sleep"
- Pairs created: 4 (1 positive, 3 random negatives)
  - ("I can't sleep", SLEEP_ISSUES) → 1 ✓
  - ("I can't sleep", FATIGUE) → 0 (random)
  - ("I can't sleep", WORTHLESSNESS) → 0 (random)
  - ("I can't sleep", COGNITIVE_ISSUES) → 0 (random)

**Class Balance:** 1:K (configurable)

**Total Samples:** N_posts × (1 + K)

**Pros:**
- Tunable class balance
- Fewer samples (faster training)
- Can focus on hard negatives

**Cons:**
- Incomplete coverage per epoch
- Noisier training signal

## Training

### Quick Start

```bash
# Train with default config (all sampling, frozen encoder)
python src/training/train_binary.py

# Train with random sampling (1:3 ratio)
python src/training/train_binary.py \
    data.negative_sampling=random \
    data.num_negatives=3

# Train with unfrozen encoder (slower, better performance)
python src/training/train_binary.py \
    model.freeze_encoder=false \
    training.batch_size=2
```

### Configuration

Edit `conf/config_binary.yaml`:

```yaml
model:
  name: google/gemma-2-2b
  pooling_strategy: mean
  freeze_encoder: true
  dropout: 0.1

training:
  num_epochs: 20
  batch_size: 8
  learning_rate: 3e-5
  use_class_weights: true  # Handle imbalance

data:
  negative_sampling: all  # or 'random'
  num_negatives: 3  # for 'random'
  max_length: 512
```

### Class Weights

With 1:9 imbalance, we use **inverse frequency weighting**:

```python
weight_positive = N_total / (2 * N_positive) ≈ 5.0
weight_negative = N_total / (2 * N_negative) ≈ 0.56
```

Loss:
```python
criterion = nn.CrossEntropyLoss(weight=[0.56, 5.0])
```

This penalizes false negatives more heavily.

## Evaluation

### Metrics

**Binary Classification Metrics:**
- **Accuracy:** Overall correctness
- **Precision:** Of predicted matches, how many are correct?
- **Recall:** Of actual matches, how many did we find?
- **F1 Score:** Harmonic mean of precision and recall
- **AUROC:** Area under ROC curve
- **AUPRC:** Area under precision-recall curve (better for imbalanced data)

**Calibration:**
- **ECE:** Expected Calibration Error
- **Reliability Diagram:** Are predicted probabilities well-calibrated?

### Inference

To classify a new post:

```python
from src.models.gemma_encoder_binary import GemmaBinaryClassifier, load_binary_tokenizer
from src.data.binary_dataset import CRITERION_DESCRIPTIONS, CRITERION_LABELS

# Load model
model = GemmaBinaryClassifier.from_pretrained('path/to/checkpoint')
tokenizer = load_binary_tokenizer('google/gemma-2-2b')

post = "I can't enjoy anything anymore"

# Test against all criteria
results = {}
for criterion_name, criterion_desc in CRITERION_DESCRIPTIONS.items():
    # Encode pair
    encoding = tokenizer(post, criterion_desc, return_tensors='pt')

    # Predict
    with torch.no_grad():
        logits = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        match_prob = probs[0, 1].item()

    results[criterion_name] = match_prob

# Top matches
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 3 matching criteria:")
for criterion, prob in sorted_results[:3]:
    print(f"  {criterion}: {prob:.3f}")
```

**Output:**
```
Top 3 matching criteria:
  ANHEDONIA: 0.892
  DEPRESSED_MOOD: 0.654
  FATIGUE: 0.421
```

## Calibration

Binary calibration is simpler than multi-class:

```python
from src.calibration.binary_calibration import (
    apply_platt_scaling,
    apply_isotonic_regression_binary,
    compute_binary_ece,
    optimize_binary_threshold
)

# Calibrate probabilities
calibrated_probs = apply_platt_scaling(uncalibrated_probs, val_labels)

# Compute ECE
ece = compute_binary_ece(calibrated_probs, val_labels)
print(f"ECE: {ece:.4f}")

# Optimize threshold
best_threshold, best_f1 = optimize_binary_threshold(
    calibrated_probs, val_labels, metric='f1'
)
print(f"Best threshold: {best_threshold:.3f}, F1: {best_f1:.3f}")
```

## Comparison: Binary vs. Multi-Class

| Aspect | Multi-Class | Binary Matching |
|--------|-------------|-----------------|
| **Task** | Which symptom? | Does post match criterion? |
| **Input** | Post only | (Post, Criterion) pair |
| **Output** | 1 of 10 classes | Binary (0 or 1) |
| **Training Samples** | N posts | N × 10 (or N × 4 with random sampling) |
| **Inference Cost** | 1 forward pass per post | 10 forward passes per post |
| **Class Balance** | Naturally imbalanced | 1:9 with 'all', tunable with 'random' |
| **Metrics** | Multi-class F1, AUPRC | Binary F1, AUROC, AUPRC |
| **Calibration** | Multi-class temperature scaling | Platt scaling, isotonic regression |

## Example Results (Expected)

With **frozen Gemma-2-2b** and **all sampling**:

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 0.88-0.92 | High due to easy negatives |
| **Precision** | 0.65-0.75 | Moderate false positives |
| **Recall** | 0.70-0.80 | Misses some matches |
| **F1 Score** | 0.67-0.77 | Harmonic mean |
| **AUROC** | 0.85-0.92 | Good discrimination |
| **AUPRC** | 0.70-0.82 | Better for imbalanced data |
| **ECE (calibrated)** | 0.05-0.10 | Well-calibrated |

**Per-Criterion Performance:**

Easier criteria (higher F1):
- DEPRESSED_MOOD, SLEEP_ISSUES, WORTHLESSNESS

Harder criteria (lower F1):
- PSYCHOMOTOR, SPECIAL_CASE, COGNITIVE_ISSUES

## Use Cases

### 1. Clinical Decision Support

```python
# Evaluate patient note against DSM-5 criteria
patient_note = "Patient reports difficulty sleeping for 2 weeks..."

# Check all criteria
matches = evaluate_all_criteria(patient_note)

# Present to clinician
print("Potential symptom matches:")
for criterion, prob in matches.items():
    if prob > 0.7:  # High confidence threshold
        print(f"  ⚠️ {criterion}: {prob:.1%}")
```

### 2. Symptom Monitoring

```python
# Track symptom evolution over time
posts_over_time = [
    ("Week 1", "I feel okay today"),
    ("Week 2", "Starting to feel down"),
    ("Week 3", "Nothing interests me anymore"),
]

for week, post in posts_over_time:
    anhedonia_score = evaluate_criterion(post, "ANHEDONIA")
    print(f"{week}: Anhedonia score = {anhedonia_score:.2f}")
```

### 3. Research: Symptom Co-occurrence

```python
# Analyze which symptoms co-occur in posts
for post in research_corpus:
    symptom_scores = evaluate_all_criteria(post)

    # Find posts with multiple symptoms
    matched_symptoms = [s for s, prob in symptom_scores.items() if prob > 0.7]

    if len(matched_symptoms) >= 5:
        print(f"Post meets criteria for Major Depressive Episode")
```

## Testing

```bash
# Run binary classification tests
pytest tests/test_binary_dataset.py -v

# Expected output:
# test_binary_dataset_creation_all_negatives PASSED
# test_binary_dataset_creation_random_negatives PASSED
# test_binary_dataset_pairs_are_correct PASSED
# test_class_imbalance_with_all_sampling PASSED
# test_class_weights_computation PASSED
# ...
```

## Future Enhancements

### 1. Hard Negative Mining

Instead of random negatives, select **hard negatives** (most confusable criteria):

```python
# Example: FATIGUE is often confused with PSYCHOMOTOR
# Prioritize these as negatives during sampling
```

### 2. Multi-Label Extension

Allow posts to match **multiple criteria** simultaneously:

```python
# Output: [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
# Post matches: ANHEDONIA, SLEEP_ISSUES, WORTHLESSNESS
```

### 3. Criterion Embeddings

Learn **dense representations** of criteria for efficient retrieval:

```python
# Encode post and all criteria
post_emb = encode_post(post)
criterion_embs = encode_all_criteria()

# Find matches via cosine similarity
similarities = cosine_similarity(post_emb, criterion_embs)
```

### 4. Uncertainty Quantification

Estimate **epistemic uncertainty** for abstention:

```python
# Use ensemble or Monte Carlo dropout
uncertainties = estimate_uncertainty(post, criterion)

if uncertainties > threshold:
    return "ABSTAIN - consult expert"
```

## References

1. **Task Design:** Similar to Natural Language Inference (NLI) and Semantic Textual Similarity (STS)
2. **Class Imbalance:** Handled via weighted loss and calibration
3. **Evaluation:** Binary classification metrics (AUROC, AUPRC, F1)
4. **Calibration:** Platt scaling, isotonic regression

---

**See also:**
- `src/data/binary_dataset.py` - Dataset implementation
- `src/models/gemma_encoder_binary.py` - Model architecture
- `src/training/train_binary.py` - Training script
- `tests/test_binary_dataset.py` - Tests

For questions or issues, see `CONTRIBUTING.md`.
