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
    "domain",
    "kx",
    "ky",
    "act",
    "lr",
    "local_epochs",
    "n_samples",
    "grid_x",
    "grid_y",
    "max_rounds",
    "time_budget",
    "seed",
    "out_root",
]
for k in keys:
    print(f"export {k.upper()}={Q(cfg[k])}")
PY
)"

# ===== 5) 诊断 =====
python - <<'PY'
import jax; print("[diag] JAX devices:", jax.devices())
from mpi4py import MPI; print("[diag] MPI vendor:", MPI.get_vendor())
PY

# ===== 6) 资源检查：优先使用 SLURM 分配列表 =====
if [ -z "${N_CLIENTS:-}" ]; then echo "[ERR] N_CLIENTS 未导出"; exit 2; fi

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
  -x NGPUS -x CUDA_VISIBLE_DEVICES \
  -x LAYERS -x DOMAIN -x KX -x KY -x ACT -x LR -x LOCAL_EPOCHS -x N_SAMPLES -x GRID_X -x GRID_Y \
  -x MAX_ROUNDS -x TIME_BUDGET -x SEED -x OUT_ROOT \
  bash -lc '
    # 绑定 GPU：优先用分配列表，否则按 nvidia-smi 数量取模
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      IFS=, read -r -a ASSIGNED <<< "${CUDA_VISIBLE_DEVICES}"
      n=${#ASSIGNED[@]}
      idx=$((OMPI_COMM_WORLD_LOCAL_RANK % n))
      export CUDA_VISIBLE_DEVICES=${ASSIGNED[$idx]}
      echo "[rank ${OMPI_COMM_WORLD_RANK}] bind GPU ${CUDA_VISIBLE_DEVICES} (from assigned list)"
    else
      export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK % NGPUS))
      echo "[rank ${OMPI_COMM_WORLD_RANK}] bind GPU ${CUDA_VISIBLE_DEVICES} / ${NGPUS} (fallback)"
    fi

    exec python -u weight_distrib_fedavg.py \
      --layers "$LAYERS" \
      --domain "$DOMAIN" \
      --kx "$KX" \
      --ky "$KY" \
      --act "$ACT" \
      --lr "$LR" \
      --local_epochs "$LOCAL_EPOCHS" \
      --n_samples "$N_SAMPLES" \
      --grid_x "$GRID_X" \
      --grid_y "$GRID_Y" \
      --max_rounds "$MAX_ROUNDS" \
      --time_budget "$TIME_BUDGET" \
      --seed "$SEED" \
      --out_root "$OUT_ROOT"
  '

