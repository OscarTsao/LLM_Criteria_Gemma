#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train -m hpo=optuna \
  train.folds=5 \
  ++hydra.sweeper.n_trials=120 \
  ++hydra.sweeper.n_jobs=4
