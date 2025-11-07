# ReDSM5 Sentence-Level Multi-Label Classification

This repository provides a reproducible pipeline for sentence-level DSM-5 criteria detection on the ReDSM5 dataset. It includes configurable encoder baselines, post-level aggregation strategies, OOF tracking, calibration utilities, and Optuna-based hyperparameter optimisation via Hydra.

## Quickstart

```bash
# 1. Environment
conda env create -f environment.yml
conda activate redsm5

# 2. Install editable package (optional)
pip install -r requirements.txt

# 3. Stage data
tree data/redsm5
# ├── redsm5_annotations.csv
# └── redsm5_posts.csv

# 4. Run a baseline (MentalBERT)
python -m src.training.train model=mentalbert

# 5. Evaluate a fold checkpoint with calibration & aggregation
python -m src.cli.run_eval ckpt_dir=outputs/<run>/fold_0 \
  calib.method=temp agg.strategy=attention
```

Add `-m hpo=optuna` to `src.training.train` for multi-run sweeps, or use the convenience scripts in `scripts/`.

## Features

- Sentence-level datasets with leak-safe post splits (`src/data/`)
- Encoder registry supporting MentalBERT, DeBERTa-V3, ModernBERT, and Gemma-2 adapters with optional LoRA
- Training loop with macro-AUPRC early stopping, AMP, gradient accumulation, and fold-wise OOF logging
- Calibration (temperature / isotonic), per-class threshold sweeps, and coverage-risk reporting
- CLI tooling for evaluation and OOF merges under `src/cli/`

## Project Layout

```
conf/                 # Hydra configs (data, model, train, sweeps)
scripts/              # Baseline & Optuna sweep wrappers
src/data/             # Dataset + split utilities
src/models/           # Registry, pooling, classifier heads
src/training/         # Train loop, losses, metrics
src/eval/             # Aggregation, calibration, reports
src/cli/              # Evaluation + OOF merge entry points
tests/                # Pytest suites (see Tests section)
```

Model checkpoints, fold metrics, and OOF artefacts are stored under `outputs/<timestamp>/fold_k/`.

## Training & HPO

- Baselines: `./scripts/run_baselines.sh`
- Canonical 5-fold: `python -m src.training.train train.folds=5`
- Optuna Sweep: `./scripts/run_sweep.sh` (configurable via `conf/hpo/optuna.yaml`)

Hydra config overrides follow `python -m src.training.train model=gemma2_encoder train.lr=3e-5 data.max_len=1024`.

## Evaluation

```
python -m src.cli.run_eval ckpt_dir=outputs/<run>/fold_0 \
  calib.method=isotonic agg.strategy=max thresholds.grid_size=2001
```

Results (metrics, per-class CSV, reliability curve, coverage-risk) are written to `<ckpt_dir>/eval/<strategy>_<calib>/`.

For global OOF analysis across runs:
```
python -m src.cli.merge_oof runs=outputs/<run1> runs=outputs/<run2> \
  output_dir=outputs/merged
```

## Testing

Install dev dependencies (`make install-dev` or `pip install -r requirements.txt`) and run:

```
pytest
```

The GitHub Action includes unit tests plus a 1-epoch CPU smoke pass to ensure import integrity.

## Safety & Clinical Use

This system is a research prototype for DSM-5 criteria detection. It must **not** be used for diagnosis, triage, or any clinical decision without qualified oversight. Refer to `MODEL_CARD.md` for limitations, escalation guidance, and abstention recommendations. Always strip PHI before logging or exporting predictions.

## Benchmarks

| Model | Macro-AUPRC | Macro-F1 | Notes |
|-------|-------------|----------|-------|
| MentalBERT | TBD | TBD | Sentence-level training |
| DeBERTa-V3 | TBD | TBD | Baseline config |
| Gemma2 + LoRA | TBD | TBD | Attention aggregation |

Update this table after confirming results in `outputs/<run>/summary.json`.
