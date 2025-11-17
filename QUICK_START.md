# Quick Start Guide

## Setup

```bash
cd /path/to/LLM_Criteria_Gemma

# Install dependencies
pip install -r requirements.txt
```

## Verify Data

```bash
ls -la data/redsm5/
# Should see:
# - redsm5_posts.csv
# - redsm5_annotations.csv
```

## Train Model

```bash
python src/training/train_gemma.py
```

This will:
1. Load the ReDSM5 dataset
2. Initialize Gemma-3-4B-IT with bidirectional attention
3. Train for 10 epochs
4. Save best model to `outputs/gemma3_baseline/best_model.pt`

## Evaluate

```bash
python src/training/evaluate.py \\
    --checkpoint outputs/gemma3_baseline/best_model.pt \\
    --split test
```

## Next Steps

See `README.md` for detailed documentation and `IMPLEMENTATION_SUMMARY.md` for technical details.
