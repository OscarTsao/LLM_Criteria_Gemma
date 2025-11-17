# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the production code: models under `src/models/`, training scripts in `src/training/`, shared helpers in `src/utils/`, and Hydra configuration glue in `conf/`.
- Hydra defaults live in `conf/config.yaml`; experiment presets (e.g., `quick_test`) sit in `conf/experiment/` and should mirror new research variants.
- Place source data in `data/redsm5/` and write all derived artifacts to `outputs/<experiment>/` so long runs stay isolated and reproducible.

## Build, Test, and Development Commands
- `make install` installs runtime deps; `make install-dev` adds pytest, black, flake8, and mypy.
- `make train-quick` (two folds, three epochs) is the staging run before `make train-5fold`, which produces `outputs/gemma3_it_5fold-google_gemma-3-4b-it-*/`.
- Evaluate checkpoints with `make evaluate CHECKPOINT=outputs/<run>/best_model.pt` or `make evaluate-best` for the canonical 5-fold result.
- Quality gates: `make lint`, `make format`, `make type-check`, and `make test` must stay green before a merge.

## Coding Style & Naming Conventions
- Target Python 3.10+, 4-space indentation, and explicit type hints on public functions; prefer dataclasses for structured configs.
- Run `black src/ --line-length=100` prior to review; flake8 is configured to ignore E203/W503 and expects descriptive module-level docstrings.
- Use lowercase, hyphenated names for new Hydra experiments (`conf/experiment/pooling-mean.yaml`) and snake_case for Python identifiers.

## Testing Guidelines
- Add pytest suites under `tests/`, mirroring the `src/` package layout (e.g., `tests/training/test_train_gemma_hydra.py`).
- Favor lightweight fixtures that reuse `data/redsm5/` samples and verify the CLI entry point (`train_gemma_hydra.py`).
- Run `make test` for the full pass; `make test-data` and `make test-models` provide quick import checks before long trainings.

## Commit & Pull Request Guidelines
- Write imperative subject lines with optional scope (`feat: extend gemma pooling options`) and keep them ≤72 characters.
- Every PR should outline motivation, linked issues, modified configs, and the exact verification commands executed.
- When training behaviour changes, attach summary metrics or log excerpts from `outputs/<experiment>/aggregate_results.json`.

## Configuration & Experiment Tips
- Override parameters inline (`python src/training/train_gemma_hydra.py training.learning_rate=3e-5`) instead of editing committed defaults.
- Promote repeatable setups to `conf/experiment/` and document their intent in `RUN_5FOLD.md` or an adjacent guide for quick discoverability.
