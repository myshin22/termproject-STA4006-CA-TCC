#!/bin/bash

###############################################################################
# NEW: Video-Level Few-Shot Experiment Runner for CA-TCC
#
# This is a NEW script. The original run_experiments.sh remains unchanged.
#
# Key differences from run_experiments.sh:
#   1. Uses main_video.py instead of main.py
#   2. Uses video-level few-shot learning (1-shot, 5-shot)
#   3. Reports video-level statistics for all splits
#   4. 1-shot = 1 video per exercise per subject (balanced across exercises)
#   5. 5-shot = 5 videos per exercise from different subjects (balanced)
#
# This script runs all experiments to verify if CA-TCC semi-supervised learning
# works better than baselines using VIDEO-LEVEL few-shot learning.
#
# Experiments:
#   1. Baseline: Random Init + 1-shot supervised (1 video/class)
#   2. Baseline: Random Init + 5-shot supervised (5 videos/class)
#   3. Baseline: Random Init + 100% supervised (upper bound)
#   4. CA-TCC (1-shot): Self-supervised → Fine-tune 1-shot
#   5. CA-TCC (5-shot): Self-supervised → Fine-tune 5-shot
#
# Usage:
#   ./run_experiments_video.sh ExerciseIMU 0 4
#   (dataset_name, start_seed, end_seed)
###############################################################################

# Check arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset_name> <start_seed> <end_seed>"
    echo "Example: $0 ExerciseIMU 0 4"
    exit 1
fi

DATASET=$1
START_SEED=$2
END_SEED=$3

EXPERIMENT_NAME="CA_TCC_VideoLevel"
DEVICE="cuda:0"

echo "=========================================================================="
echo "CA-TCC Video-Level Few-Shot Experiments"
echo "=========================================================================="
echo "Dataset: $DATASET"
echo "Seeds: $START_SEED to $END_SEED"
echo "Device: $DEVICE"
echo ""
echo "NOTE: This uses VIDEO-LEVEL few-shot learning:"
echo "  - 1-shot = 1 video per exercise class"
echo "  - 5-shot = 5 videos per exercise class (from different subjects)"
echo "=========================================================================="
echo ""

# Function to run experiment
run_experiment() {
    local run_name=$1
    local seed=$2
    local training_mode=$3

    echo ""
    echo "----------------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: $run_name (seed=$seed, mode=$training_mode)"
    echo "----------------------------------------------------------------------"

    # CHANGED: Use main_video.py instead of main.py
    python main_video.py \
        --experiment_description "$EXPERIMENT_NAME" \
        --run_description "$run_name" \
        --seed $seed \
        --selected_dataset $DATASET \
        --training_mode $training_mode \
        --device $DEVICE

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $run_name (seed=$seed)"
    else
        echo "✗ Failed: $run_name (seed=$seed)"
    fi
}

# Main experiment loop
for seed in $(seq $START_SEED $END_SEED); do

    echo ""
    echo "=========================================================================="
    echo "SEED $seed / $END_SEED"
    echo "=========================================================================="

    # -------------------------------------------------------------------------
    # BASELINES: Random Initialization (video-level few-shot)
    # -------------------------------------------------------------------------

    echo ""
    echo ">>> BASELINES (Random Initialization - Video-Level Few-Shot)"

    # Baseline 1: 1-shot supervised only (1 video per class)
    run_experiment "Baseline_1shot" $seed "supervised_1shot"

    # Baseline 2: 5-shot supervised only (5 videos per class)
    run_experiment "Baseline_5shot" $seed "supervised_5shot"

    # Baseline 3: 100% supervised (upper bound)
    run_experiment "Baseline_100p" $seed "supervised"

    # -------------------------------------------------------------------------
    # CA-TCC: Self-supervised Pretraining + Fine-tuning (video-level)
    # -------------------------------------------------------------------------

    echo ""
    echo ">>> CA-TCC METHODS (Video-Level Few-Shot)"

    # CA-TCC Step 1: Self-supervised pretraining on ALL data (no labels)
    run_experiment "CATCC" $seed "self_supervised"

    # CA-TCC (1-shot): Fine-tune with 1-shot labeled data (1 video/class)
    run_experiment "CATCC" $seed "ft_1shot"

    # CA-TCC (5-shot): Fine-tune with 5-shot labeled data (5 videos/class from different subjects)
    run_experiment "CATCC" $seed "ft_5shot"

    # Note: For full CA-TCC pipeline with pseudo-labeling:
    # Uncomment these if you want to run the complete semi-supervised pipeline
    # run_experiment "CATCC" $seed "gen_pseudo_labels"  # Generate pseudo labels
    # run_experiment "CATCC" $seed "SupCon"              # Supervised contrastive with pseudo labels
    # run_experiment "CATCC" $seed "ft_SupCon_1shot"     # Final fine-tuning

done

echo ""
echo "=========================================================================="
echo "✓ All experiments completed!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. Analyze results:"
echo "     python compare_results.py --experiment_name $EXPERIMENT_NAME --dataset $DATASET"
echo ""
echo "  2. Check logs:"
echo "     ls experiments_logs/$EXPERIMENT_NAME/"
echo ""
