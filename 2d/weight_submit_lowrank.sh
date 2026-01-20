#!/bin/bash
# 写配置(JSON) -> 串行 sbatch 多个单节点作业 -> 作业内用 mpirun 起 N_CLIENTS 进程（lowrank 2D + fixed local training）
set -euo pipefail

############ 实验配置（可用环境变量覆盖） ############
LAYERS="${LAYERS:-[2,20,20,20,20,1]}"
# 2D 区域和剖分
DOMAIN="${DOMAIN:-[0,1,0,1]}"
KX="${KX:-3}"  # x 方向均分块数
KY="${KY:-3}"  # y 方向均分块数
ACT="${ACT:-tanh}"
LR="${LR:-1e-4}"
N_SAMPLES="${N_SAMPLES:-1000}"
GRID_X="${GRID_X:-256}"  # x 方向测试网格点数
GRID_Y="${GRID_Y:-256}"  # y 方向测试网格点数
MAX_ROUNDS="${MAX_ROUNDS:-10000}"

# 关键超参：支持单值或多值（空格/逗号/括号均可）
LOCAL_EPOCHS="${LOCAL_EPOCHS:-20 200}"
STEP_SIZE="${STEP_SIZE:-0.5}"
TOP_K="${TOP_K:-20}"

PINV_TOL="${PINV_TOL:-1e-5}"

# 时间预算（秒）
TIME_BUDGET="${TIME_BUDGET:-300}"

# 可选
SEED="${SEED:-0}"
ALG_TAG="${ALG_TAG:-lowrank-2d}"

# 计算 N_CLIENTS（需要在OUT_ROOT之前计算）
N_CLIENTS=$((KX * KY))

# 输出路径包含客户端数量和网格信息
OUT_ROOT="${OUT_ROOT:-Results_lowrank_2d_K${KX}x${KY}}"

############ 资源（单节点多卡） ############
PARTITION="${PARTITION:-general}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
REQUEST_GPUS="${REQUEST_GPUS:-7}"

# N_CLIENTS already computed above
if [[ "$KX" -le 0 ]] || [[ "$KY" -le 0 ]]; then
  echo "[ERR] KX 和 KY 必须大于0" >&2; exit 1
fi

TOTAL_SAMPLES=$(( N_CLIENTS * N_SAMPLES ))
echo "[plan] single-node: clients=$N_CLIENTS (${KX}x${KY}); request_gpus=$REQUEST_GPUS; per-client N_SAMPLES=$N_SAMPLES; total_samples=$TOTAL_SAMPLES; tb=${TIME_BUDGET}s"

############ 生成通用标签 ############
ARCH_TAG=$(python - <<PY
import ast
layers = ast.literal_eval("""$LAYERS""")
print("-".join(str(x) for x in layers))
PY
)
DOMAIN_HASH=$(python - <<PY
import ast, hashlib, json
domain = ast.literal_eval("""$DOMAIN""")
print(hashlib.sha1(json.dumps(domain, sort_keys=True).encode()).hexdigest()[:6])
PY
)

STAMP_GLOBAL=$(date +%Y%m%d_%H%M%S)
CFG_DIR="$PWD/cfg"
mkdir -p "$CFG_DIR"

############ 规范化列表：支持 a b c / a,b,c / (a,b,c) ############
_norm_list() {
  local s="$1"
  s="${s//,/ }"; s="${s//\(/}"; s="${s//\)/}"
  echo "$s" | awk '{$1=$1; print}'
}
EPOCHS_LIST=$(_norm_list "$LOCAL_EPOCHS")
STEP_LIST=$(_norm_list "$STEP_SIZE")
TOP_K_LIST=$(_norm_list "$TOP_K")

echo "[grid] LOCAL_EPOCHS=($EPOCHS_LIST)"
echo "[grid] STEP_SIZE=($STEP_LIST)"
echo "[grid] TOP_K=($TOP_K_LIST)"
echo "[const] PINV_TOL=$PINV_TOL"

############ 串行提交（笛卡尔积） ############
idx=0
for EPOCHS in $EPOCHS_LIST; do
  for STEP in $STEP_LIST; do
    for K in $TOP_K_LIST; do
          idx=$((idx+1))
          STAMP="${STAMP_GLOBAL}_i${idx}"

          CFG_FILE="$CFG_DIR/${ALG_TAG}_cfg_arch_${ARCH_TAG}_K${KX}x${KY}_D${DOMAIN_HASH}_epochs${EPOCHS}_step${STEP}_k${K}_rmax${MAX_ROUNDS}_tb${TIME_BUDGET}s_seed${SEED}_${STAMP}.json"

          python - <<PY
import json, ast
cfg = {
  "layers": ast.literal_eval("""$LAYERS"""),
  "domain": ast.literal_eval("""$DOMAIN"""),
  "kx": int($KX),
  "ky": int($KY),
  "act": "$ACT",
  "lr": float($LR),
  "local_epochs": int($EPOCHS),
  "n_samples": int($N_SAMPLES),
  "grid_x": int($GRID_X),
  "grid_y": int($GRID_Y),
  "max_rounds": int($MAX_ROUNDS),
  "pinv_tol": float($PINV_TOL),
  "step_size": float($STEP),
  "top_k": int($K),
  "time_budget": float($TIME_BUDGET),
  "seed": int($SEED),
  "out_root": "$OUT_ROOT"
}
json.dump(cfg, open("$CFG_FILE","w"))
print("Wrote", "$CFG_FILE")
PY

          SBATCH_ARGS=(
            --nodes=1
            --gres=gpu:${REQUEST_GPUS}
            --cpus-per-task="${CPUS_PER_TASK}"
            --mem="${MEM}"
            -t "${TIME_LIMIT}"
            -J "${ALG_TAG}_1n${N_CLIENTS}_K${KX}x${KY}_g${REQUEST_GPUS}_epochs${EPOCHS}_step${STEP}_k${K}_tb${TIME_BUDGET}s"
            -p "${PARTITION}"
            -o "slurm-%A.out"
          )
          EXPORTS="ALL,N_CLIENTS=$N_CLIENTS,CFG_FILE=$CFG_FILE,REQUEST_GPUS=$REQUEST_GPUS"

          echo "[submit] #$idx  sbatch ${SBATCH_ARGS[*]}"
          sbatch "${SBATCH_ARGS[@]}" --export="$EXPORTS" weight_run_lowrank.sh

          sleep 0.2
    done
    done
done

echo "[done] submitted ${idx} jobs."

