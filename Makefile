# Makefile for Gemma Encoder on ReDSM5
# Usage: make <target>

.PHONY: help install test clean train train-5fold train-5fold-mentallama train-5fold-gemma train-5fold-both train-quick evaluate lint format check-gpu nli-test nli-quick nli-train nli-simple nli-predict-interactive nli-predict-demo nli-predict-best

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies
	pip install -r requirements.txt

install-viz: ## Install visualization dependencies only
	pip install rich>=13.0.0 plotext>=5.2.0

install-dev: ## Install with development dependencies
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

##@ Training

train: ## Train with original script (single split)
	python src/training/train_gemma.py

train-5fold: train-5fold-mentallama ## Alias: run default 5-fold training (MentaLLaMA)

train-5fold-mentallama: ## Train 5-fold CV with MentaLLaMA-chat-7B encoder
	python src/training/train_gemma_hydra.py output.experiment_name=mentallama_5fold

train-5fold-gemma: ## Train 5-fold CV with Gemma-2 (unfrozen encoder)
	python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b model.freeze_encoder=false training.batch_size=2 output.experiment_name=gemma_5fold

train-5fold-both: ## Train 5-fold CV sequentially for MentaLLaMA then Gemma
	@$(MAKE) train-5fold-mentallama
	@$(MAKE) train-5fold-gemma

train-quick: ## Quick test (2 folds, 3 epochs)
	python src/training/train_gemma_hydra.py experiment=quick_test

train-gemma9b: ## Train with Gemma-9B model
	python src/training/train_gemma_hydra.py model.name=google/gemma-2-9b training.batch_size=8

train-attention: ## Train with attention pooling
	python src/training/train_gemma_hydra.py model.pooling_strategy=attention

train-10fold: ## Train with 10-fold CV
	python src/training/train_gemma_hydra.py cv.num_folds=10

##@ NLI Binary Classification

nli-test: ## Test NLI dataset creation and text-pair formatting
	python scripts/test_nli_dataset.py

nli-quick: ## Quick NLI test (2 folds, 3 epochs)
	python src/training/train_nli_5fold.py experiment=nli_quick_test

nli-train: ## Full NLI 5-fold CV training (production)
	python src/training/train_nli_5fold.py experiment=nli_full_5fold

nli-imbalanced: ## NLI with imbalanced data (3:1 negative:positive)
	python src/training/train_nli_5fold.py experiment=nli_imbalanced

nli-simple: ## Simple NLI training (single split, no CV)
	python src/training/train_nli_simple.py \
		--data_dir data/redsm5 \
		--model_name google/gemma-2-2b \
		--batch_size 8 \
		--epochs 10 \
		--output_dir outputs/nli_simple \
		--freeze_encoder

nli-gemma-2b: ## NLI 5-fold CV with Gemma-2B
	python src/training/train_nli_5fold.py \
		model.name=google/gemma-2-2b \
		training.batch_size=8 \
		output.experiment_name=nli_gemma2b_5fold

nli-gemma-9b: ## NLI 5-fold CV with Gemma-9B
	python src/training/train_nli_5fold.py \
		model.name=google/gemma-2-9b \
		training.batch_size=4 \
		output.experiment_name=nli_gemma9b_5fold

nli-short-criteria: ## NLI with short criterion descriptions
	python src/training/train_nli_5fold.py \
		data.use_short_criteria=true \
		output.experiment_name=nli_short_criteria

nli-full-criteria: ## NLI with full criterion descriptions
	python src/training/train_nli_5fold.py \
		data.use_short_criteria=false \
		output.experiment_name=nli_full_criteria

nli-unfreeze: ## NLI with unfrozen encoder (full fine-tuning)
	python src/training/train_nli_5fold.py \
		model.freeze_encoder=false \
		training.batch_size=2 \
		training.num_epochs=10 \
		output.experiment_name=nli_unfrozen

nli-pooling-mean: ## NLI with mean pooling
	python src/training/train_nli_5fold.py \
		model.pooling_strategy=mean \
		output.experiment_name=nli_pooling_mean

nli-pooling-attention: ## NLI with attention pooling
	python src/training/train_nli_5fold.py \
		model.pooling_strategy=attention \
		output.experiment_name=nli_pooling_attention

nli-ablation-pooling: ## NLI pooling strategy ablation study
	@echo "Running pooling ablation study for NLI..."
	@for pooler in mean cls max attention; do \
		echo "Training NLI with $$pooler pooling..."; \
		python src/training/train_nli_5fold.py \
			model.pooling_strategy=$$pooler \
			output.experiment_name=nli_ablation_$$pooler \
			experiment=nli_quick_test; \
	done
	@echo "NLI pooling ablation complete. Check outputs/nli_ablation_*/"

nli-ablation-negatives: ## NLI negative ratio ablation study
	@echo "Running negative ratio ablation study..."
	@for ratio in 0.5 1.0 2.0 3.0; do \
		echo "Training NLI with negative_ratio=$$ratio..."; \
		python src/training/train_nli_5fold.py \
			data.negative_ratio=$$ratio \
			output.experiment_name=nli_neg_$$ratio \
			experiment=nli_quick_test; \
	done
	@echo "Negative ratio ablation complete. Check outputs/nli_neg_*/"

nli-show-results: ## Show NLI aggregate results from latest run
	@python -c "import json; \
		from pathlib import Path; \
		import glob; \
		results_files = sorted(glob.glob('outputs/nli_*/aggregate_results.json')); \
		if results_files: \
			latest = results_files[-1]; \
			print(f'Latest NLI results: {latest}'); \
			with open(latest) as f: \
				r = json.load(f); \
				print(f'\n5-Fold CV Results:'); \
				print(f'  Mean F1:  {r[\"mean_f1\"]:.4f} ± {r[\"std_f1\"]:.4f}'); \
				print(f'  Mean AUC: {r[\"mean_auc\"]:.4f} ± {r[\"std_auc\"]:.4f}'); \
				print(f'\nPer-Fold Results:'); \
				for fold in r['fold_results']: \
					print(f'  Fold {fold[\"fold\"]}: F1={fold[\"best_val_f1\"]:.4f}, AUC={fold[\"best_val_auc\"]:.4f}'); \
		else: \
			print('No NLI results found. Run make nli-train first.')"

nli-quickstart: ## Complete NLI quick start workflow
	@echo "═══════════════════════════════════════"
	@echo "NLI Binary Classification Quick Start"
	@echo "═══════════════════════════════════════"
	@echo ""
	@echo "Step 1: Testing NLI dataset creation..."
	@$(MAKE) nli-test
	@echo ""
	@echo "Step 2: Running quick training (2 folds, 3 epochs)..."
	@$(MAKE) nli-quick
	@echo ""
	@echo "Step 3: Showing results..."
	@$(MAKE) nli-show-results
	@echo ""
	@echo "✓ Quick start complete!"
	@echo "  For full training: make nli-train"
	@echo "  For documentation: cat README_NLI.md"

##@ Inference & Prediction

nli-predict-interactive: ## Interactive NLI prediction (requires CHECKPOINT)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT not specified"; \
		echo "Usage: make nli-predict-interactive CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	python src/inference/predict_nli.py --checkpoint $(CHECKPOINT) --mode interactive

nli-predict-demo: ## Run prediction demo with examples (requires CHECKPOINT)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT not specified"; \
		echo "Usage: make nli-predict-demo CHECKPOINT=path/to/model.pt"; \
		exit 1; \
	fi
	python src/inference/predict_nli.py --checkpoint $(CHECKPOINT) --mode demo

nli-predict-batch: ## Batch prediction from file (requires CHECKPOINT, FILE, CRITERION)
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(FILE)" ] || [ -z "$(CRITERION)" ]; then \
		echo "Error: Missing required arguments"; \
		echo "Usage: make nli-predict-batch CHECKPOINT=path/to/model.pt FILE=posts.txt CRITERION=DEPRESSED_MOOD"; \
		exit 1; \
	fi
	python src/inference/predict_nli.py --checkpoint $(CHECKPOINT) --mode batch --post_file $(FILE) --criterion $(CRITERION)

nli-predict-best: ## Interactive prediction with best model from latest run
	@RUN_DIR=$$(for dir in $$(ls -td outputs/nli_*/ 2>/dev/null); do \
		if [ -f "$${dir}fold_1_best.pt" ]; then echo $$dir; fi; \
	done | head -n 1); \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No trained NLI model found. Run 'make nli-train' first."; \
		exit 1; \
	fi; \
	CHECKPOINT="$${RUN_DIR%/}/fold_1_best.pt"; \
	echo "Using checkpoint: $$CHECKPOINT"; \
	python src/inference/predict_nli.py --checkpoint "$$CHECKPOINT" --mode interactive

nli-demo-best: ## Run prediction demo with best model from latest run
	@RUN_DIR=$$(for dir in $$(ls -td outputs/nli_*/ 2>/dev/null); do \
		if [ -f "$${dir}fold_1_best.pt" ]; then echo $$dir; fi; \
	done | head -n 1); \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No trained NLI model found. Run 'make nli-train' first."; \
		exit 1; \
	fi; \
	CHECKPOINT="$${RUN_DIR%/}/fold_1_best.pt"; \
	echo "Using checkpoint: $$CHECKPOINT"; \
	python src/inference/predict_nli.py --checkpoint "$$CHECKPOINT" --mode demo

##@ Evaluation

evaluate: ## Evaluate trained model (requires --checkpoint argument)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT not specified"; \
		echo "Usage: make evaluate CHECKPOINT=path/to/checkpoint.pt"; \
		exit 1; \
	fi
	python src/training/evaluate.py --checkpoint $(CHECKPOINT)

evaluate-best: ## Evaluate best model from a 5-fold run (override with RUN=outputs/<run_dir>)
	@if [ -z "$(RUN)" ]; then \
		RUN_DIR=$$(for dir in $$(ls -td outputs/*/ 2>/dev/null); do \
			if [ -f "$${dir}fold_0/best_model.pt" ]; then echo $$dir; fi; \
		done | head -n 1); \
	else \
		RUN_DIR="$(RUN)"; \
	fi; \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No run directory found. Run 'make train-5fold' first."; \
		exit 1; \
	fi; \
	if [ ! -f "$${RUN_DIR%/}/fold_0/best_model.pt" ]; then \
		echo "Best checkpoint not found in $$RUN_DIR/fold_0/"; \
		exit 1; \
	fi; \
	CHECKPOINT="$${RUN_DIR%/}/fold_0/best_model.pt"; \
	echo "Evaluating checkpoint $$CHECKPOINT"; \
	python src/training/evaluate.py --checkpoint "$$CHECKPOINT"

show-results: ## Show aggregate 5-fold results for latest run (override with RUN=outputs/<run_dir>)
	@if [ -z "$(RUN)" ]; then \
		RUN_DIR=$$(for dir in $$(ls -td outputs/*/ 2>/dev/null); do \
			if [ -f "$${dir}aggregate_results.json" ]; then echo $$dir; fi; \
		done | head -n 1); \
	else \
		RUN_DIR="$(RUN)"; \
	fi; \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No run directory found. Run 'make train-5fold' first."; \
		exit 1; \
	fi; \
	if [ ! -f "$${RUN_DIR%/}/aggregate_results.json" ]; then \
		echo "aggregate_results.json not found in $$RUN_DIR"; \
		exit 1; \
	fi; \
	echo "Showing aggregate results from $$RUN_DIR"; \
	cat "$${RUN_DIR%/}/aggregate_results.json" | python -m json.tool

##@ Data

prepare-splits: ## Create CV splits manually
	python -c "from src.data.cv_splits import create_cv_splits; \
		create_cv_splits('data/redsm5/redsm5_annotations.csv', num_folds=5, output_dir='data/redsm5/cv_splits')"

check-data: ## Verify dataset files exist
	@echo "Checking dataset files..."
	@if [ -f data/redsm5/redsm5_posts.csv ]; then \
		echo "✓ redsm5_posts.csv found"; \
	else \
		echo "✗ redsm5_posts.csv missing"; \
	fi
	@if [ -f data/redsm5/redsm5_annotations.csv ]; then \
		echo "✓ redsm5_annotations.csv found"; \
		wc -l data/redsm5/redsm5_annotations.csv; \
	else \
		echo "✗ redsm5_annotations.csv missing"; \
	fi

data-stats: ## Show dataset statistics
	python -c "import pandas as pd; \
		df = pd.read_csv('data/redsm5/redsm5_annotations.csv'); \
		print(f'Total samples: {len(df)}'); \
		print(f'\nSymptom distribution:\n{df[\"DSM5_symptom\"].value_counts()}'); \
		print(f'\nStatus distribution:\n{df[\"status\"].value_counts()}')"

##@ Code Quality

lint: ## Run linting checks
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format: ## Format code with black
	black src/ --line-length=100

format-check: ## Check code formatting without changes
	black src/ --check --line-length=100

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports

##@ Testing

test: ## Run all tests
	pytest tests/ -v

test-models: ## Test model imports
	python -c "from src.models import GemmaClassifier, GemmaEncoder; print('✓ Model imports OK')"

test-data: ## Test data loading
	python -c "from src.data import load_redsm5; print('✓ Data imports OK')"

test-imports: ## Test all imports
	python -c "from src.models import GemmaClassifier; \
		from src.data import load_redsm5; \
		print('✓ All imports OK')"

##@ Cleanup

clean: ## Remove generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .mypy_cache
	@echo "Cleaned up generated files"

clean-outputs: ## Remove training outputs (BE CAREFUL!)
	@echo "WARNING: This will delete all training outputs!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	rm -rf outputs/
	rm -rf data/redsm5/cv_splits/
	@echo "Outputs cleaned"

clean-all: clean clean-outputs ## Remove all generated files and outputs

##@ GPU & System

check-gpu: ## Check GPU availability and memory
	@echo "GPU Information:"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); exec('if torch.cuda.is_available():\\n for i in range(torch.cuda.device_count()):\\n  print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\")\\n  print(f\"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB\")')"

check-env: ## Check Python environment
	@echo "Python Environment:"
	@python --version
	@echo "\nInstalled packages:"
	@pip list | grep -E "(torch|transformers|hydra|pandas|numpy|scikit-learn)" || echo "Key packages not found"

##@ Documentation

docs: ## Open documentation in browser
	@echo "Opening documentation..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open README.md; \
	elif command -v open > /dev/null; then \
		open README.md; \
	else \
		echo "Please open README.md manually"; \
	fi

show-config: ## Show current Hydra configuration
	python src/training/train_gemma_hydra.py --cfg job

##@ Monitoring

tensorboard: ## Launch TensorBoard (if logs exist)
	@if [ -d outputs/tensorboard ]; then \
		tensorboard --logdir outputs/tensorboard; \
	else \
		echo "No TensorBoard logs found"; \
	fi

watch-training: ## Watch training logs in real-time
	@if [ -f outputs/training.log ]; then \
		tail -f outputs/training.log; \
	else \
		echo "No training log found"; \
	fi

##@ Experiments

exp-pooling-comparison: ## Compare all pooling strategies
	@echo "Running pooling strategy comparison..."
	@for pooler in mean cls max attention; do \
		echo "Training with $$pooler pooling..."; \
		python src/training/train_gemma_hydra.py \
			model.pooling_strategy=$$pooler \
			output.experiment_name=pooling_$$pooler \
			experiment=quick_test; \
	done
	@echo "Comparison complete. Check outputs/pooling_*/"

exp-learning-rates: ## Test different learning rates
	@echo "Testing learning rates..."
	@for lr in 1e-5 2e-5 3e-5 5e-5; do \
		echo "Training with LR=$$lr..."; \
		python src/training/train_gemma_hydra.py \
			training.learning_rate=$$lr \
			output.experiment_name=lr_$$lr \
			experiment=quick_test; \
	done
	@echo "LR experiments complete. Check outputs/lr_*/"

##@ Quick Commands

quick-check: check-data test-imports check-gpu ## Quick sanity check of setup

full-pipeline: install check-data train-5fold show-results ## Full training pipeline

demo: train-quick show-results ## Quick demo (2 folds, 3 epochs)

##@ Information

info: ## Show project information
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  Gemma Encoder for DSM-5 Criteria Matching (ReDSM5)       ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Paper: arXiv:2503.02656 (Gemma Encoder)"
	@echo "Dataset: ReDSM5 (arXiv:2508.03399)"
	@echo ""
	@echo "Key Features:"
	@echo "  • Bidirectional attention (decoder → encoder)"
	@echo "  • 6 pooling strategies (mean, cls, max, attention, etc.)"
	@echo "  • 5-fold cross-validation with Hydra"
	@echo "  • Two task modes: Multi-class & NLI Binary"
	@echo ""
	@echo "Task 1: Original Multi-Class Classification"
	@echo "  Input:  Single text (post)"
	@echo "  Output: Symptom class (0-9)"
	@echo "  F1:     0.72-0.75"
	@echo ""
	@echo "Task 2: NLI Binary Criteria Matching (NEW)"
	@echo "  Input:  [CLS] post [SEP] criterion [SEP]"
	@echo "  Output: Binary (matched/unmatched)"
	@echo "  F1:     0.70-0.80"
	@echo ""
	@echo "Quick Start - Original Task:"
	@echo "  make install          # Install dependencies"
	@echo "  make train-quick      # Quick test (30 min)"
	@echo "  make train-5fold      # Full 5-fold CV (2-3 hours)"
	@echo ""
	@echo "Quick Start - NLI Task:"
	@echo "  make nli-quickstart   # Complete NLI workflow"
	@echo "  make nli-quick        # Quick NLI test"
	@echo "  make nli-train        # Full NLI 5-fold CV"
	@echo ""
	@echo "Documentation:"
	@echo "  README.md             # Original task"
	@echo "  README_NLI.md         # NLI task"
	@echo "  make help             # All commands"

version: ## Show version information
	@echo "Project: Gemma Encoder for ReDSM5"
	@echo "Version: 0.1.0"
	@grep -E "^__version__" src/__init__.py 2>/dev/null || echo "Version: Not set"

##@ NLI Documentation

nli-docs: ## Show NLI documentation
	@cat README_NLI.md

nli-summary: ## Show NLI refactoring summary
	@cat REFACTORING_SUMMARY.md

##@ Quick Reference

.PHONY: ref
ref: ## Quick reference card
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  GEMMA ENCODER - QUICK REFERENCE"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "SETUP:"
	@echo "  make install          Install dependencies"
	@echo "  make check-data       Verify dataset files"
	@echo "  make check-gpu        Check GPU availability"
	@echo ""
	@echo "ORIGINAL TASK (Multi-class):"
	@echo "  make train-quick      Quick test (2 folds)"
	@echo "  make train-5fold      Full 5-fold CV"
	@echo "  make show-results     Show latest results"
	@echo ""
	@echo "NLI TASK (Binary):"
	@echo "  make nli-test         Test NLI dataset"
	@echo "  make nli-quick        Quick NLI (2 folds)"
	@echo "  make nli-train        Full NLI 5-fold CV"
	@echo "  make nli-show-results Show NLI results"
	@echo "  make nli-quickstart   Complete NLI workflow"
	@echo ""
	@echo "NLI VARIATIONS:"
	@echo "  make nli-gemma-2b     NLI with Gemma-2B"
	@echo "  make nli-gemma-9b     NLI with Gemma-9B"
	@echo "  make nli-imbalanced   Imbalanced data (3:1)"
	@echo "  make nli-unfreeze     Unfreeze encoder"
	@echo ""
	@echo "INFERENCE:"
	@echo "  make nli-predict-best       Interactive prediction (latest model)"
	@echo "  make nli-demo-best          Demo with examples (latest model)"
	@echo "  make nli-predict-interactive CHECKPOINT=path/to/model.pt"
	@echo "  make nli-predict-demo       CHECKPOINT=path/to/model.pt"
	@echo ""
	@echo "EXPERIMENTS:"
	@echo "  make nli-ablation-pooling   Pooling strategies"
	@echo "  make nli-ablation-negatives Negative ratios"
	@echo "  make exp-pooling-comparison Original pooling"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make clean            Clean cache files"
	@echo "  make test             Run tests"
	@echo "  make format           Format code"
	@echo "  make info             Project info"
	@echo "  make help             All commands"
	@echo "═══════════════════════════════════════════════════════════"
