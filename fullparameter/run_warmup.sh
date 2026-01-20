#!/bin/bash
#SBATCH --job-name=warmup_fedavg
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:5
#SBATCH --time=72:00:00

set -euo pipefail

export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH

source activate jax124

# Configuration (aligned with CIFAR-10 ResNet-20 FedAvg baseline)
TOTAL_CLIENTS=100
CLIENT_FRACTION=0.05

# Hyperparameters (aligned with CIFAR-10 ResNet-20 FedAvg baseline)
WARMUP_ROUNDS_VALUES=(1200)
ALPHA_VALUES=(0.6 0.3 0.1 0.05 0.01)
INITIAL_LR_VALUES=(0.1)  # Aligned with train_fedavg_only.py (default=0.1)
LOCAL_EPOCHS=(5)

# Model sizes to iterate over
MODEL_SIZES=("light" "medium20k" "large200k" "resnet20")

# Checkpoint settings
CHECKPOINT_INTERVAL=100  # Save checkpoint every N rounds

# Model-specific settings (aligned with FedRCL)
EVAL_BATCH_SIZE=128  # Aligned with FedRCL
TRAIN_BATCH_SIZE=50  # Aligned with FedRCL
WEIGHT_DECAY=1e-3  # Aligned with FedRCL
LR_DECAY=0.998  # Aligned with FedRCL (per round)

OUTPUT_BASE="warmup_checkpoints"

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "${PROJECT_DIR}"
WARMUP_SCRIPT="${PROJECT_DIR}/warmup_fedavg.py"

mkdir -p "${OUTPUT_BASE}"

# Setup GPUs
if [ -n "${SLURM_JOB_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES=$(echo "${SLURM_JOB_GPUS}" | tr -d '[]')
    echo "Slurm assigned GPUs: ${CUDA_VISIBLE_DEVICES}"
else
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    echo "Using default GPUs: ${CUDA_VISIBLE_DEVICES}"
fi

IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS=${#GPU_LIST[@]}

if [ ${NUM_GPUS} -lt 1 ]; then
    echo "Error: No GPUs available"
    exit 1
fi

echo "Total clients: ${TOTAL_CLIENTS}, GPUs available: ${NUM_GPUS}"
echo "Client fraction per round: ${CLIENT_FRACTION} (activates $(python3 -c "print(max(1, int(${TOTAL_CLIENTS} * ${CLIENT_FRACTION})))") clients)"
echo "Warmup rounds search space: ${WARMUP_ROUNDS_VALUES[*]}"
echo "Models to train: ${MODEL_SIZES[*]}"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL} rounds"

CONFIG_COUNT=0
TOTAL=$((${#WARMUP_ROUNDS_VALUES[@]} * ${#ALPHA_VALUES[@]} * ${#INITIAL_LR_VALUES[@]} * ${#LOCAL_EPOCHS[@]} * ${#MODEL_SIZES[@]}))

echo "Total configurations: ${TOTAL}"

for model_size in "${MODEL_SIZES[@]}"; do
  for warmup_rounds in "${WARMUP_ROUNDS_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
      for initial_lr in "${INITIAL_LR_VALUES[@]}"; do
        for local_epochs in "${LOCAL_EPOCHS[@]}"; do
          
          CONFIG_COUNT=$((CONFIG_COUNT + 1))
          
          # Create checkpoint directory for this configuration
          checkpoint_dir="${OUTPUT_BASE}/model_${model_size}_a${alpha}_lr${initial_lr}_le${local_epochs}_w${warmup_rounds}"
          mkdir -p "${checkpoint_dir}"
          
          run_name="warmup_${model_size}_a${alpha}_lr${initial_lr}_le${local_epochs}_w${warmup_rounds}"
          
          echo "========================================="
          echo "Config ${CONFIG_COUNT}/${TOTAL}"
          echo "Model: ${model_size}"
          echo "alpha=${alpha}, lr=${initial_lr}, local_epochs=${local_epochs}"
          echo "warmup_rounds=${warmup_rounds}"
          echo "checkpoint_dir=${checkpoint_dir}"
          echo "========================================="
          
          mpirun --bind-to none --map-by slot \
            -x CUDA_VISIBLE_DEVICES \
            -x XLA_PYTHON_CLIENT_PREALLOCATE=false \
            -x TF_CPP_MIN_LOG_LEVEL=2 \
            -x XLA_FLAGS \
            --mca btl_vader_single_copy_mechanism none \
            --mca orte_base_help_aggregate 0 \
            --mca btl_base_warn_component_unused 0 \
            --mca mca_base_component_show_load_errors 0 \
            -np 5 python "${WARMUP_SCRIPT}" \
            --total_clients ${TOTAL_CLIENTS} \
            --client_fraction ${CLIENT_FRACTION} \
            --warmup_rounds ${warmup_rounds} \
            --alpha ${alpha} \
            --initial_lr ${initial_lr} \
            --local_epochs ${local_epochs} \
            --model_size ${model_size} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --train_batch_size ${TRAIN_BATCH_SIZE} \
            --weight_decay ${WEIGHT_DECAY} \
            --lr_decay ${LR_DECAY} \
            --checkpoint_dir "${checkpoint_dir}" \
            --checkpoint_interval ${CHECKPOINT_INTERVAL} \
            --client_selection_seed 0 \
            2>&1 | tee "${checkpoint_dir}/warmup_training.log"
          
          echo "Config ${CONFIG_COUNT}/${TOTAL} completed"
          echo "Checkpoints saved in: ${checkpoint_dir}/"
          echo "----------------------------------------"
        done
      done
    done
  done
done

echo "=========================================="
echo "All warmup configurations completed!"
echo "Checkpoints saved in: ${OUTPUT_BASE}/"
echo "=========================================="

