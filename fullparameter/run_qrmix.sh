#!/bin/bash
#SBATCH --job-name=qr_mix_cifar10
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:5
#SBATCH --time=48:00:00

set -euo pipefail

export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH

source activate jax124

# ===================================================================
# Configuration for QR-mix from checkpoints
# ===================================================================

# Basic settings
TOTAL_CLIENTS=100
CLIENT_FRACTION=0.05
QR_ROUNDS=20  # Run 200 rounds of QR-mix

# Checkpoint settings
CHECKPOINT_BASE="warmup_checkpoints"
CHECKPOINT_ROUND_VALUES=(200 400 600 800)  # Checkpoint rounds to test (can specify multiple values)

# Model architectures to test
MODEL_SIZES=("resnet20" "large200k" "medium20k" "light")

# Alpha (heterogeneity) values to test
ALPHA_VALUES=(0.01 0.05)

# Training hyperparameters (will be auto-loaded from checkpoint if available)
INITIAL_LR=0.1
LOCAL_EPOCHS=5
LR_DECAY=0.998
LR_SCALE=0.1  # Learning rate scaling factor (divide LR by 10)

# Subspace parameters
RANK_K_VALUES=(200)
OVERSAMPLING_VALUES=(50)
POWER_ITER_VALUES=(9)
BLOCK_SIZE_VALUES=(32)

# Regularization parameters (beta, gamma)
BETA_GAMMA_PAIRS=("0.5 0.5")

# Other settings
EVAL_BATCH_SIZE=128
TRAIN_BATCH_SIZE=50
WEIGHT_DECAY=1e-3

OUTPUT_BASE="qr_mix_results_from_ckpt"

# ===================================================================
# Setup
# ===================================================================

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "${PROJECT_DIR}"
TRAIN_SCRIPT="${PROJECT_DIR}/main.py"

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

echo "=========================================="
echo "QR-mix from Checkpoint Experiment"
echo "=========================================="
echo "Total clients: ${TOTAL_CLIENTS}, GPUs available: ${NUM_GPUS}"
echo "Client fraction per round: ${CLIENT_FRACTION}"
echo "Checkpoint rounds: ${CHECKPOINT_ROUND_VALUES[*]}"
echo "QR-mix rounds: ${QR_ROUNDS}"
echo "Models: ${MODEL_SIZES[*]}"
echo "Alpha values: ${ALPHA_VALUES[*]}"
echo ""

# Calculate total configurations
TOTAL=$((${#CHECKPOINT_ROUND_VALUES[@]} * ${#MODEL_SIZES[@]} * ${#ALPHA_VALUES[@]} * ${#BETA_GAMMA_PAIRS[@]} * ${#RANK_K_VALUES[@]} * ${#OVERSAMPLING_VALUES[@]} * ${#POWER_ITER_VALUES[@]} * ${#BLOCK_SIZE_VALUES[@]}))
echo "Total configurations: ${TOTAL}"
echo "=========================================="
echo ""

CONFIG_COUNT=0

# ===================================================================
# Main loop: iterate over all configurations
# ===================================================================

for checkpoint_round in "${CHECKPOINT_ROUND_VALUES[@]}"; do
  # Create output directory for this checkpoint round
  output_base_round="${OUTPUT_BASE}${checkpoint_round}"
  mkdir -p "${output_base_round}"
  
  for model_size in "${MODEL_SIZES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
      for pair in "${BETA_GAMMA_PAIRS[@]}"; do
        IFS=' ' read -r beta gamma <<< "${pair}"
        for rank_k in "${RANK_K_VALUES[@]}"; do
          for oversampling in "${OVERSAMPLING_VALUES[@]}"; do
            for power_iter in "${POWER_ITER_VALUES[@]}"; do
              for block_size in "${BLOCK_SIZE_VALUES[@]}"; do
                
                CONFIG_COUNT=$((CONFIG_COUNT + 1))
                
                # Construct checkpoint path
                # Expected format: warmup_checkpoints/model_{model_size}_a{alpha}_lr{lr}_le{le}_w{warmup}/checkpoint_round_{round}.pkl
                checkpoint_dir="${CHECKPOINT_BASE}/model_${model_size}_a${alpha}_lr${INITIAL_LR}_le${LOCAL_EPOCHS}_w1200"
                checkpoint_path="${checkpoint_dir}/checkpoint_round_${checkpoint_round}.pkl"
                
                # Check if checkpoint exists
                if [ ! -f "${checkpoint_path}" ]; then
                  echo "========================================="
                  echo "Config ${CONFIG_COUNT}/${TOTAL} - SKIPPED"
                  echo "Model: ${model_size}, Alpha: ${alpha}, Checkpoint Round: ${checkpoint_round}"
                  echo "Checkpoint not found: ${checkpoint_path}"
                  echo "========================================="
                  echo ""
                  continue
                fi
                
                # Create output directory
                run_name="qrmix_${model_size}_a${alpha}_ckpt${checkpoint_round}_qr${QR_ROUNDS}_k${rank_k}_s${oversampling}_q${power_iter}_bs${block_size}_beta${beta}_gamma${gamma}"
                run_dir="${output_base_round}/${run_name}"
                mkdir -p "${run_dir}"
                
                echo "========================================="
                echo "Config ${CONFIG_COUNT}/${TOTAL}"
                echo "Model: ${model_size}"
                echo "Alpha (heterogeneity): ${alpha}"
                echo "Checkpoint Round: ${checkpoint_round}"
                echo "Checkpoint: ${checkpoint_path}"
                echo "QR-mix rounds: ${QR_ROUNDS} (from round ${checkpoint_round})"
                echo "Subspace: k=${rank_k}, s=${oversampling}, q=${power_iter}, block_size=${block_size}"
                echo "Regularization: beta=${beta}, gamma=${gamma}"
                echo "Output: ${run_dir}"
                echo "========================================="
                
                # Run QR-mix training from checkpoint
                mpirun --bind-to none --map-by slot \
                  -x CUDA_VISIBLE_DEVICES \
                  -x XLA_PYTHON_CLIENT_PREALLOCATE=false \
                  -x TF_CPP_MIN_LOG_LEVEL=2 \
                  -x XLA_FLAGS \
                  --mca btl_vader_single_copy_mechanism none \
                  --mca orte_base_help_aggregate 0 \
                  --mca btl_base_warn_component_unused 0 \
                  --mca mca_base_component_show_load_errors 0 \
                  -np 5 python "${TRAIN_SCRIPT}" \
                  --checkpoint_path "${checkpoint_path}" \
                  --checkpoint_round ${checkpoint_round} \
                  --qr_rounds ${QR_ROUNDS} \
                  --alpha ${alpha} \
                  --model_size ${model_size} \
                  --total_clients ${TOTAL_CLIENTS} \
                  --client_fraction ${CLIENT_FRACTION} \
                  --local_epochs ${LOCAL_EPOCHS} \
                  --rank_k ${rank_k} \
                  --oversampling ${oversampling} \
                  --power_iter ${power_iter} \
                  --block_size ${block_size} \
                  --beta ${beta} \
                  --gamma ${gamma} \
                  --eval_batch_size ${EVAL_BATCH_SIZE} \
                  --train_batch_size ${TRAIN_BATCH_SIZE} \
                  --weight_decay ${WEIGHT_DECAY} \
                  --initial_lr ${INITIAL_LR} \
                  --lr_decay ${LR_DECAY} \
                  --lr_scale ${LR_SCALE} \
                  --output_dir "${run_dir}" \
                  2>&1 | tee "${run_dir}/training.log"
                
                echo ""
                echo "Config ${CONFIG_COUNT}/${TOTAL} completed"
                echo "Results saved to: ${run_dir}/training_results.csv"
                echo "----------------------------------------"
                echo ""
                
              done
            done
          done
        done
      done
    done
  done
done

echo "=========================================="
echo "All configurations completed!"
echo "Total configs run: ${CONFIG_COUNT}/${TOTAL}"
echo "Results saved in: ${OUTPUT_BASE}*/"
echo "=========================================="
