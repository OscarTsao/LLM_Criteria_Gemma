#!/bin/bash
# Run baseline comparisons across different models and configurations
#
# Usage: bash scripts/run_baselines.sh [--quick]
#
# Runs experiments with:
# - Different models (MentaLBERT, DeBERTa-v3, Gemma-2-2b)
# - Different pooling strategies
# - Different max lengths
#
# Results saved to: results/baselines.csv

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in quick mode (1 epoch, small batches)"
fi

# Create output directory
mkdir -p results

# Function to run experiment
run_experiment() {
    local model=$1
    local pooling=$2
    local max_len=$3
    local exp_name="${model//\//_}_${pooling}_len${max_len}"

    echo "=========================================="
    echo "Running: $exp_name"
    echo "=========================================="

    if [ "$QUICK_MODE" = true ]; then
        # Quick mode: 1 epoch, 1 fold
        python src/training/train_gemma_hydra.py \
            model.name="$model" \
            model.pooling_strategy="$pooling" \
            data.max_length="$max_len" \
            training.num_epochs=1 \
            training.batch_size=2 \
            cv.enabled=false \
            output.experiment_name="baseline_${exp_name}_quick" \
            hydra.run.dir="outputs/baselines/${exp_name}_quick"
    else
        # Full mode: 5-fold CV
        python src/training/train_gemma_hydra.py \
            model.name="$model" \
            model.pooling_strategy="$pooling" \
            data.max_length="$max_len" \
            cv.enabled=true \
            cv.num_folds=5 \
            output.experiment_name="baseline_${exp_name}" \
            hydra.run.dir="outputs/baselines/${exp_name}"
    fi

    echo ""
}

# === BASELINE MODELS ===

# 1. MentaLBERT (if available)
# run_experiment "mental/mental-bert-base-uncased" "mean" 512

# 2. DeBERTa-v3
# run_experiment "microsoft/deberta-v3-base" "mean" 512

# 3. Gemma-2-2b with different pooling
run_experiment "google/gemma-2-2b" "mean" 512
run_experiment "google/gemma-2-2b" "cls" 512
run_experiment "google/gemma-2-2b" "attention" 512

# 4. Different sequence lengths
run_experiment "google/gemma-2-2b" "mean" 256
run_experiment "google/gemma-2-2b" "mean" 1024

echo "=========================================="
echo "Baselines complete!"
echo "=========================================="

# Aggregate results
if [ "$QUICK_MODE" = false ]; then
    echo "Aggregating results..."
    python scripts/aggregate_results.py \
        --experiment-dir outputs/baselines \
        --output results/baselines.csv

    echo "Results saved to: results/baselines.csv"
    echo ""
    echo "To view:"
    echo "  cat results/baselines.csv"
    echo "  python scripts/plot_baselines.py"
fi
