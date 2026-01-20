#!/bin/bash
#SBATCH -o slurm-%A.out
set -euo pipefail

# ===== 1) 环境 =====
source ~/.bashrc
source activate jax124

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=1

export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# 可选稳定性设置
export CUBLAS_WORKSPACE_CONFIG=:16:8
export NVIDIA_TF32_OVERRIDE=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0

# ===== 2) CUDA12 动态库路径 & 软链接 =====
PIP_NVIDIA_LIBS="$(
python - <<'PY'
import site,glob,os
paths=[]
for base in site.getsitepackages():
    paths+=glob.glob(os.path.join(base,'nvidia','*','lib'))
print(':'.join(paths))
PY
)"
if [ -n "${PIP_NVIDIA_LIBS}" ]; then
  export LD_LIBRARY_PATH="${PIP_NVIDIA_LIBS}:${LD_LIBRARY_PATH-}"
  IFS=':' read -r -a _arr <<< "${PIP_NVIDIA_LIBS}"
  for d in "${_arr[@]}"; do
    [ -e "${d}/libcusparse.so.12" ] && [ ! -e "${d}/libcusparse.so" ] && ln -sfn libcusparse.so.12 "${d}/libcusparse.so"
    for lib in cublas cudnn cusolver curand cufft nvJitLink; do
      [ -e "${d}/lib${lib}.so.12" ] && [ ! -e "${d}/lib${lib}.so" ] && ln -sfn "lib${lib}.so.12" "${d}/lib${lib}.so"
    done
  done
fi

echo "[diag] python: $(which python)"
echo "[diag] CUDA_VISIBLE_DEVICES(initial): ${CUDA_VISIBLE_DEVICES-}"
echo

# ===== 3) 配置文件 =====
if [ -n "${CFG_LIST:-}" ]; then
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[ERR] CFG_LIST 已提供，但 SLURM_ARRAY_TASK_ID 未设置" >&2; exit 2
  fi
  CFG_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CFG_LIST")
fi

if [ -z "${CFG_FILE:-}" ] || [ ! -f "$CFG_FILE" ]; then
  echo "[ERR] CFG_FILE 未设置或不存在：$CFG_FILE" >&2; exit 2
fi
echo "[diag] using CFG_FILE=$CFG_FILE"

# ===== 4) 读取 JSON -> 导出为环境变量 =====
eval "$(
python - <<'PY'
import os, json, shlex
cfg = json.load(open(os.environ["CFG_FILE"]))
def Q(v):
    import json as _j
    if isinstance(v, (list, dict)): v = _j.dumps(v)
    return shlex.quote(str(v))
keys = [
    "layers",
    "edges",
    "act",
    "lr",
    "local_epochs",
    "n_samples",
    "n_boundary",
    "lambda_bc",
    "lambda_bratu",
    "grid",
    "max_rounds",
    "pinv_tol",
    "step_size",
    "top_k",
    "time_budget",
    "out_root",
]
for k in keys:
    print(f"export {k.upper()}={Q(cfg[k])}")
PY
)"

# ===== 5) 诊断 =====
# 注意：不在mpirun之前初始化JAX，因为此时CUDA_VISIBLE_DEVICES还未正确设置
# JAX设备诊断将在每个MPI进程内部执行（在设置CUDA_VISIBLE_DEVICES之后）
python - <<'PY'
from mpi4py import MPI; print("[diag] MPI vendor:", MPI.get_vendor())
PY

# ===== 6) 资源检查：优先使用 SLURM 分配列表 =====
if [ -z "${N_CLIENTS:-}" ]; then echo "[ERR] N_CLIENTS 未导出"; exit 2; fi

# 设置默认GPU池（固定使用GPU 3,4,5,6,7）
# 注意：SLURM的--export无法正确传递包含逗号的环境变量值
# 所以在这里直接设置默认值，忽略环境变量传递的值
ALLOWED_GPUS="${ALLOWED_GPUS:-3,4,5,6,7}"
# 检查是否只传递了一个值（不包含逗号），如果是则使用默认值
if [ "${ALLOWED_GPUS}" = "${ALLOWED_GPUS%,*}" ] && [ "${ALLOWED_GPUS}" != "3,4,5,6,7" ]; then
  ALLOWED_GPUS="3,4,5,6,7"
  echo "[diag] ALLOWED_GPUS传递失败（只收到一个值），使用默认GPU池: ${ALLOWED_GPUS}"
else
  echo "[diag] GPU池限制: ALLOWED_GPUS=${ALLOWED_GPUS}"
fi
export ALLOWED_GPUS

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a ASSIGNED <<< "${CUDA_VISIBLE_DEVICES}"
  NGPUS=${#ASSIGNED[@]}
  echo "[diag] assigned_gpus=${ASSIGNED[*]} ; NGPUS=$NGPUS (from env)"
else
  NGPUS=$(nvidia-smi -L | wc -l || echo 0)
  echo "[diag] NGPUS=$NGPUS (from nvidia-smi)"
fi
if [ "$NGPUS" -le 0 ]; then echo "[ERR] 未检测到可用 GPU"; exit 2; fi
export NGPUS

if [ "$N_CLIENTS" -gt "$NGPUS" ]; then
  echo "[note] N_CLIENTS($N_CLIENTS) > NGPUS($NGPUS)：将做 GPU 共享映射"
fi

# ===== 7) mpirun 本地起 N_CLIENTS 个进程 =====
HOST="${SLURMD_NODENAME:-$HOSTNAME}"

mpirun -np "${N_CLIENTS}" \
  --host "${HOST}" \
  --oversubscribe \
  --bind-to none --map-by slot \
  -x PATH -x LD_LIBRARY_PATH -x CONDA_PREFIX \
  -x JAX_PLATFORMS -x XLA_PYTHON_CLIENT_PREALLOCATE -x TF_CPP_MIN_LOG_LEVEL \
  -x OMPI_MCA_btl -x OMPI_MCA_pml -x OMPI_MCA_btl_vader_single_copy_mechanism \
  -x NGPUS -x ALLOWED_GPUS \
  -x LAYERS -x EDGES -x ACT -x LR -x LOCAL_EPOCHS -x N_SAMPLES -x N_BOUNDARY -x LAMBDA_BC -x LAMBDA_BRATU \
  -x GRID -x MAX_ROUNDS -x STEP_SIZE -x TOP_K -x TIME_BUDGET -x OUT_ROOT \
  -x PINV_TOL \
  bash -c '
    # 在JAX初始化之前设置CUDA_VISIBLE_DEVICES
    # 使用固定的GPU池（例如 3,4,5,6,7）
    IFS=, read -r -a GPU_POOL <<< "${ALLOWED_GPUS}"
    n=${#GPU_POOL[@]}
    rank_idx=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    gpu_idx=$((rank_idx % n))
    selected_gpu=${GPU_POOL[$gpu_idx]}
    
    # 设置CUDA_VISIBLE_DEVICES为单个GPU，这样JAX只会看到这个GPU
    export CUDA_VISIBLE_DEVICES=${selected_gpu}
    echo "[rank ${OMPI_COMM_WORLD_RANK}] bind GPU ${selected_gpu} (from ALLOWED_GPUS pool, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
    
    # 现在JAX初始化时只会看到这个GPU（在Python代码中初始化）
    exec python -u pinn_bratu.py \
      --layers "$LAYERS" \
      --edges "$EDGES" \
      --act "$ACT" \
      --lr "$LR" \
      --local_epochs "$LOCAL_EPOCHS" \
      --n_samples "$N_SAMPLES" \
      --n_boundary "$N_BOUNDARY" \
      --lambda_bc "$LAMBDA_BC" \
      --lambda_bratu "$LAMBDA_BRATU" \
      --grid "$GRID" \
      --max_rounds "$MAX_ROUNDS" \
      --pinv_tol "$PINV_TOL" \
      --step_size "$STEP_SIZE" \
      --top_k "$TOP_K" \
      --time_budget "$TIME_BUDGET" \
      --out_root "$OUT_ROOT"
  '

