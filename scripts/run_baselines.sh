#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train model=mentalbert
python -m src.training.train model=deberta_v3_base
python -m src.training.train model=gemma2_encoder
