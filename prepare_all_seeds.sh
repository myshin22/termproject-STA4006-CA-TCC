#!/bin/bash
#
# Prepare CA-TCC data for all seeds
#
# This creates separate data folders for each seed
# Each seed will have different train/val/test subject splits
#

SEEDS=(0 1 2 3 4)

echo "======================================================================"
echo "Preparing CA-TCC data for all seeds"
echo "======================================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Preparing CA-TCC data for seed ${SEED}..."
    echo "----------------------------------------------------------------------"

    OUTPUT_DIR="/home/user1/MY_project/proj_ca_tcc/CA-TCC/data/ExerciseIMU_seed${SEED}"

    python prepare_exercise_data_video_v2.py \
        --seed ${SEED} \
        --output_dir ${OUTPUT_DIR}

    if [ $? -ne 0 ]; then
        echo "ERROR: CA-TCC data preparation failed for seed ${SEED}"
        exit 1
    fi

    echo "✓ Completed seed ${SEED}"
done

echo ""
echo "======================================================================"
echo "All CA-TCC data preparation complete!"
echo "======================================================================"
echo ""
echo "Created directories:"
for SEED in "${SEEDS[@]}"; do
    echo "  - ExerciseIMU_seed${SEED}/"
done
