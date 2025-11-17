# Makefile for Gemma Encoder on ReDSM5
# Usage: make <target>

.PHONY: help install test clean train train-5fold train-quick evaluate lint format check-gpu

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install with development dependencies
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

##@ Training

train: ## Train with default configuration
	python src/training/train_gemma_hydra.py

train-5fold: ## Train 5-fold CV with google/gemma-3-4b-it encoder
	python src/training/train_gemma_hydra.py model.name=google/gemma-3-4b-it output.experiment_name=gemma3_it_5fold

train-quick: ## Quick test (2 folds, 3 epochs)
	python src/training/train_gemma_hydra.py experiment=quick_test

train-attention: ## Train with attention pooling
	python src/training/train_gemma_hydra.py model.pooling_strategy=attention

train-10fold: ## Train with 10-fold CV
	python src/training/train_gemma_hydra.py cv.num_folds=10

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
	@echo "  • Expected F1: 0.72-0.75"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install          # Install dependencies"
	@echo "  make train-quick      # Quick test (30 min)"
	@echo "  make train-5fold      # Full 5-fold CV (2-3 hours)"
	@echo ""
	@echo "For more info: make help"

version: ## Show version information
	@echo "Project: Gemma Encoder for ReDSM5"
	@echo "Version: 0.1.0"
	@grep -E "^__version__" src/__init__.py 2>/dev/null || echo "Version: Not set"
