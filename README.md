# Gemma Encoder for DSM-5 Criteria Matching

Implementation of the Gemma Encoder architecture from the paper ["Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks"](https://arxiv.org/abs/2503.02656) applied to DSM-5 mental health criteria matching using the ReDSM5 dataset.

## Overview

This project implements best methods from the Gemma Encoder paper for criteria matching tasks on the ReDSM5 dataset:
- **Bidirectional attention** adaptation of Gemma decoder models
- **Multiple pooling strategies** (Mean, First-K, Last-K, Attention-based)
- **Hyperparameter optimization** for dropout rates and architecture
- **GLUE-style evaluation metrics** for classification performance
- **Binary NLI criteria matching** for interpretable post-criterion matching (NEW!)

### Two Task Formulations

This project supports two complementary approaches:

1. **Multi-class Classification** (Original): Classify each post into one of 10 symptom categories
2. **Binary NLI Criteria Matching** (NEW): Binary classification for post-criterion pairs
   - Input: `[CLS] post [SEP] criterion [SEP]`
   - Output: `matched` or `unmatched`
   - Creates 1,484 posts × 9 criteria = 13,356 training examples
   - See [Binary NLI Documentation](docs/BINARY_NLI_CRITERIA_MATCHING.md) for details

## Key Features

### 1. Bidirectional Attention
The critical innovation: converting Gemma's causal (unidirectional) attention to bidirectional attention dramatically improves performance on encoding tasks.

```python
from src.models.gemma_encoder import GemmaClassifier

model = GemmaClassifier(
    num_classes=10,
    model_name="google/gemma-3-4b-it",
    pooling_strategy="mean"
)
```

### 2. Multiple Pooling Strategies
Implemented poolers from the paper:
- **Mean Pooling**: Average over all tokens
- **First-K Pooling**: Aggregate first K tokens
- **Last-K Pooling**: Aggregate last K tokens
- **Attention Pooling (KV)**: Learnable query over key-value pairs
- **Attention Pooling (Query)**: Multi-head probe attention

### 3. ReDSM5 Dataset
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
- SPECIAL_CASE (expert discrimination cases)

## Installation

```bash
# Clone the repository (if applicable)
cd /path/to/LLM_Criteria_Gemma

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Multi-class Classification (Original)

1. **Prepare Data**: Ensure ReDSM5 dataset is in `data/redsm5/`
2. **Train Model**:
   ```bash
   python src/training/train_gemma_hydra.py
   ```
3. **Evaluate**:
   ```bash
   python src/training/evaluate.py --checkpoint outputs/gemma3_baseline/best_model.pt
   ```

### Binary NLI Criteria Matching (NEW)

1. **Prepare Data**: Same as above
2. **Train Binary NLI Model**:
   ```bash
   python src/training/train_nli_binary.py
   ```
3. **Evaluate**:
   ```bash
   python src/training/evaluate_nli_binary.py --checkpoint outputs/nli_binary-*/best_model.pt
   ```
4. **Test Dataset**:
   ```bash
   python tests/test_nli_dataset.py
   ```

See [Binary NLI Documentation](docs/BINARY_NLI_CRITERIA_MATCHING.md) for comprehensive guide.

## Project Structure

```
LLM_Criteria_Gemma/
├── data/
│   ├── DSM5/
│   │   └── MDD_Criteria.json     # DSM-5 criteria (placeholder)
│   └── redsm5/
│       ├── redsm5_posts.csv
│       └── redsm5_annotations.csv
├── src/
│   ├── models/
│   │   ├── poolers.py            # Pooling strategies
│   │   ├── gemma_encoder.py      # Bidirectional Gemma encoder
│   │   └── dora.py               # DoRA parameter-efficient fine-tuning
│   ├── data/
│   │   ├── dsm5_criteria.py      # DSM-5 criterion descriptions (NEW)
│   │   ├── redsm5_nli_dataset.py # Binary NLI dataset loader (NEW)
│   │   ├── redsm5_dataset.py     # Original multi-class dataset
│   │   └── cv_splits.py          # Cross-validation utilities
│   ├── training/
│   │   ├── train_nli_binary.py   # Binary NLI training (NEW)
│   │   ├── evaluate_nli_binary.py # Binary NLI evaluation (NEW)
│   │   ├── train_gemma_hydra.py  # Multi-class training with CV
│   │   └── evaluate.py           # Multi-class evaluation
│   └── utils/
│       └── logger.py             # Logging utilities
├── tests/
│   ├── test_nli_dataset.py       # Test binary NLI dataset (NEW)
│   └── test_*.py                 # Other tests
├── docs/
│   └── BINARY_NLI_CRITERIA_MATCHING.md  # Binary NLI guide (NEW)
├── conf/
│   ├── config.yaml               # Hydra defaults
│   └── experiment/
│       └── quick_test.yaml       # Example overrides
├── outputs/                       # Training outputs
├── requirements.txt
└── README.md
```

## Configuration

Edit `conf/config.yaml` (Hydra defaults) or add overrides under `conf/experiment/` to customize:
- Model checkpoint (`google/gemma-3-4b-it` by default)
- Pooling strategy
- Hyperparameters (learning rate, dropout, batch size)
- Data splits and CV settings

## Model Options

The project now targets the Gemma-3 family. Override `model.name` to explore other checkpoints.

| Model | Parameters | GPU Memory* | Notes |
|-------|-----------|-------------|-------|
| google/gemma-3-4b-it | 4B | ~16GB | Default instruct-tuned encoder-friendly choice |
| google/gemma-3-12b-it | 12B | ~48GB | Higher accuracy, requires larger GPUs |

<sub>*Approximate memory when fine-tuning with bfloat16.</sub>

## Results

Expected performance on ReDSM5 criteria matching:
- **Accuracy**: 70-80%
- **Macro F1**: 0.65-0.75
- **Per-class F1**: Varies by symptom prevalence

Comparison with baselines (from DataAug_Criteria_Evidence project):
- BERT baseline: F1 ~0.65
- RoBERTa baseline: F1 ~0.70
- **Gemma-3-4b-it (ours)**: F1 ~0.72-0.75

## Key Implementation Details

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
pooler = MeanPooler(hidden_dim=2048)

# Attention pooling with learnable query
pooler = AttentionPoolerKV(hidden_dim=2048)
```

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
