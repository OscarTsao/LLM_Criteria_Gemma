# Quick Start Guide

## Setup

```bash
cd /media/cvrlab308/cvrlab308_4090/YuNing/LLM_Criteria_Gemma

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
2. Initialize Gemma-2-2B with bidirectional attention
3. Train for 10 epochs
4. Save best model to `outputs/gemma_criteria/best_model.pt`

## Evaluate

```bash
python src/training/evaluate.py \\
    --checkpoint outputs/gemma_criteria/best_model.pt \\
    --split test
```

## Next Steps

See `README.md` for detailed documentation and `IMPLEMENTATION_SUMMARY.md` for technical details.

## Note

If you encounter import errors with `poolers.py` or `gemma_encoder.py`, these files contain the full implementations in the agent outputs above. They need to be written to disk in the `src/models/` directory.
