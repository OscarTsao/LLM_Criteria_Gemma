# Update Log - Hydra + 5-Fold CV

## Changes Made

### ✅ Added Hydra Configuration Management
- Created `conf/config.yaml` - Main Hydra configuration
- Created `conf/experiment/quick_test.yaml` - Quick test config
- Created `conf/experiment/full_5fold.yaml` - Full 5-fold config
- Updated `requirements.txt` - Added hydra-core and omegaconf

### ✅ Implemented 5-Fold Cross-Validation
- Created `src/data/cv_splits.py` - CV split utilities
  - `create_cv_splits()` - Stratified K-fold splitting
  - `get_fold_statistics()` - Fold statistics
  - `load_fold_split()` - Load specific fold
  
- Created `src/training/train_gemma_hydra.py` - Main training script
  - `FoldTrainer` class for single fold training
  - Automatic 5-fold CV execution
  - Aggregate results computation

### ✅ New Documentation
- Created `HYDRA_GUIDE.md` - Complete Hydra usage guide
- Created `RUN_5FOLD.md` - Step-by-step 5-fold CV guide
- Created `UPDATE_LOG.md` - This file

## Usage

### Basic 5-Fold Training
```bash
python src/training/train_gemma_hydra.py
```

### With Parameter Overrides
```bash
python src/training/train_gemma_hydra.py \
    model.name=google/gemma-3-12b-it \
    training.batch_size=2 \
    cv.num_folds=10
```

### Quick Test
```bash
python src/training/train_gemma_hydra.py experiment=quick_test
```

## Features

✅ **Hydra Configuration**
- Composable configs
- Command-line overrides
- Experiment management
- Automatic output organization

✅ **5-Fold Cross-Validation**
- Stratified splits by DSM5 symptom
- Per-fold model checkpoints
- Aggregate statistics (mean, std, min, max F1)
- Reproducible with random seed

✅ **Output Organization**
```
outputs/
└── experiment_name/
    ├── fold_0/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    ├── fold_4/
    ├── cv_results.csv
    └── aggregate_results.json
```

## Comparison

### Before (Original)
- Single train/val/test split
- Manual configuration
- No CV support
- Simple training loop

### After (Hydra + 5-Fold)
- 5-fold stratified CV
- Hydra configuration management
- Aggregate statistics

## Benefits

1. **Robust Evaluation**: 5-fold CV reduces variance in performance estimates
2. **Easy Experimentation**: Override any parameter from command line
3. **Reproducibility**: Config saved with each experiment
4. **Organization**: Automatic output directory structure
5. **Flexibility**: Multiple experiment configs

## Backward Compatibility

✅ Original training script still works:
```bash
python src/training/train_gemma.py
```

✅ All existing code unchanged:
- `src/models/` - No changes
- `src/data/redsm5_dataset.py` - No changes (extended with cv_splits.py)

## Next Steps

- [ ] Add ensemble prediction from all 5 folds
- [ ] Implement Hydra multirun for hyperparameter sweeps
- [ ] Add test set evaluation after CV

---
**Date**: November 5, 2025
**Status**: Production-ready with 5-fold CV support
