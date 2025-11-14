#!/bin/bash
#
# Run CA-TCC Few-Shot Learning Experiments
#
# This script runs:
# 1. Self-supervised pretraining (on train+val, NO test!)
# 2. 0-shot evaluation (evaluate pretrained model, NO fine-tuning)
# 3. 1-shot fine-tuning (pretrain + 1-shot)
# 4. 5-shot fine-tuning (pretrain + 5-shot)
# 5. Supervised 1-shot baseline (train from scratch with 1-shot)
# 6. Supervised 5-shot baseline (train from scratch with 5-shot)
#
# For each of 5 seeds
#

GPU=0
SEEDS=(0 1 2 3 4)
DATASET="ExerciseIMU"
EXPERIMENT_DESC="FewShot_ExerciseIMU"

echo "======================================================================"
echo "CA-TCC Few-Shot Learning Experiments"
echo "======================================================================"
echo "GPU: $GPU"
echo "Dataset: $DATASET"
echo "Seeds: ${SEEDS[@]}"
echo ""

cd "$(dirname "$0")"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Seed ${SEED}"
    echo "======================================================================"

    RUN_DESC="seed_${SEED}"
    DATA_PATH="data"
    # main_video.py appends dataset name to data_path, so we modify dataset name
    DATASET_SEED="${DATASET}_seed${SEED}"

    # Check if data exists
    if [ ! -d "${DATA_PATH}/${DATASET_SEED}" ]; then
        echo "ERROR: Data directory not found: ${DATA_PATH}/${DATASET_SEED}"
        echo "Please run prepare_all_seeds.sh first"
        exit 1
    fi

    # Step 1: Self-supervised pretraining
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 1/6: Self-supervised pretraining"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode self_supervised \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU \
        2>&1 | tee -a "logs_seed${SEED}_pretrain.log"

    if [ $? -ne 0 ]; then
        echo "ERROR: Pretraining failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ Pretraining completed for seed ${SEED}"

    # Step 2: 0-shot evaluation (NO fine-tuning)
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 2/6: 0-shot evaluation (pretrained model, NO fine-tuning)"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode 0shot \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU

    if [ $? -ne 0 ]; then
        echo "ERROR: 0-shot failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ 0-shot completed for seed ${SEED}"

    # Step 3: 1-shot fine-tuning (pretrain + 1-shot)
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 3/6: 1-shot fine-tuning (pretrain + 1-shot)"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode ft_1shot \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU

    if [ $? -ne 0 ]; then
        echo "ERROR: 1-shot fine-tuning failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ 1-shot fine-tuning completed for seed ${SEED}"

    # Step 4: 5-shot fine-tuning (pretrain + 5-shot)
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 4/6: 5-shot fine-tuning (pretrain + 5-shot)"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode ft_5shot \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU

    if [ $? -ne 0 ]; then
        echo "ERROR: 5-shot fine-tuning failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ 5-shot fine-tuning completed for seed ${SEED}"

    # Step 5: Supervised 1-shot baseline (train from scratch)
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 5/6: Supervised 1-shot baseline (train from scratch)"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode supervised_1shot \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU

    if [ $? -ne 0 ]; then
        echo "ERROR: Supervised 1-shot baseline failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ Supervised 1-shot baseline completed for seed ${SEED}"

    # Step 6: Supervised 5-shot baseline (train from scratch)
    echo ""
    echo "----------------------------------------------------------------------"
    echo "[${SEED}] Step 6/6: Supervised 5-shot baseline (train from scratch)"
    echo "----------------------------------------------------------------------"

    python main_video.py \
        --experiment_description $EXPERIMENT_DESC \
        --run_description $RUN_DESC \
        --seed $SEED \
        --training_mode supervised_5shot \
        --selected_dataset $DATASET_SEED \
        --data_path $DATA_PATH \
        --device cuda:$GPU

    if [ $? -ne 0 ]; then
        echo "ERROR: Supervised 5-shot baseline failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ Supervised 5-shot baseline completed for seed ${SEED}"

    echo ""
    echo "======================================================================"
    echo "✓ All experiments completed for seed ${SEED}"
    echo "======================================================================"
done

echo ""
echo "======================================================================"
echo "✓✓✓ ALL EXPERIMENTS COMPLETED! ✓✓✓"
echo "======================================================================"
echo ""
echo "Results saved in: experiments_logs/${EXPERIMENT_DESC}/${RUN_DESC}/"
echo ""
echo "Next steps:"
echo "  1. Aggregate results: python compare_results_video.py"
echo "  2. Check logs in experiments_logs/"
