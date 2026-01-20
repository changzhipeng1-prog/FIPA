#!/bin/bash
# 写配置(JSON) -> 串行 sbatch 多个单节点作业 -> 作业内用 mpirun 起 N_CLIENTS 进程（FedAvg + fixed local training）
set -euo pipefail

############ 实验配置（可用环境变量覆盖） ############
LAYERS="${LAYERS:-[1,20,20,20,1]}"
EDGES="${EDGES:-[0,0.1,0.3,0.5,0.6,0.7,0.85,0.9,1.0]}"  # 8 clients
# EDGES="${EDGES:-[0,0.5,1.0]}"  # 2 clients
# EDGES="${EDGES:-[0,0.25,0.5,0.75,1.0]}"  # 4 clients
ACT="${ACT:-tanh}"
N_FREQ="${N_FREQ:-8}"
LR="${LR:-1e-4}"
N_SAMPLES="${N_SAMPLES:-400}"
GRID="${GRID:-1024}"
MAX_ROUNDS="${MAX_ROUNDS:-10000}"

# 关键超参：支持单值或多值（空格/逗号/括号均可）
LOCAL_EPOCHS="${LOCAL_EPOCHS:-20 1000}"

# 时间预算（秒）
TIME_BUDGET="${TIME_BUDGET:-30}"

# 可选
SEED="${SEED:-0}"
ALG_TAG="${ALG_TAG:-fedavg}"

# 计算 N_CLIENTS（需要在OUT_ROOT之前计算）
N_CLIENTS=$(python - <<PY
import ast
edges = ast.literal_eval("""$EDGES""")
print(len(edges)-1)
PY
)

# 输出路径包含客户端数量和N_FREQ
OUT_ROOT="${OUT_ROOT:-Results_fedavg_N${N_CLIENTS}_nf${N_FREQ}}"

############ 资源（单节点多卡） ############
PARTITION="${PARTITION:-general}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
REQUEST_GPUS="${REQUEST_GPUS:-7}"

# N_CLIENTS already computed above
if [[ "$N_CLIENTS" -le 0 ]]; then
  echo "[ERR] EDGES 无效（长度<=1）" >&2; exit 1
fi

TOTAL_SAMPLES=$(( N_CLIENTS * N_SAMPLES ))
echo "[plan] single-node: clients=$N_CLIENTS; request_gpus=$REQUEST_GPUS; per-client N_SAMPLES=$N_SAMPLES; total_samples=$TOTAL_SAMPLES; tb=${TIME_BUDGET}s"

############ 生成通用标签 ############
ARCH_TAG=$(python - <<PY
import ast
layers = ast.literal_eval("""$LAYERS""")
print("-".join(str(x) for x in layers))
PY
)
EDGE_HASH=$(python - <<PY
import ast, hashlib, json
edges = ast.literal_eval("""$EDGES""")
print(hashlib.sha1(json.dumps(edges, sort_keys=True).encode()).hexdigest()[:6])
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

echo "[grid] LOCAL_EPOCHS=($EPOCHS_LIST)"

############ 串行提交 ############
idx=0
for EPOCHS in $EPOCHS_LIST; do
          idx=$((idx+1))
          STAMP="${STAMP_GLOBAL}_i${idx}"

          CFG_FILE="$CFG_DIR/${ALG_TAG}_cfg_arch_${ARCH_TAG}_N${N_CLIENTS}_E${EDGE_HASH}_nf${N_FREQ}_epochs${EPOCHS}_rmax${MAX_ROUNDS}_tb${TIME_BUDGET}s_seed${SEED}_${STAMP}.json"

          python - <<PY
import json, ast
cfg = {
  "layers": ast.literal_eval("""$LAYERS"""),
  "edges":  ast.literal_eval("""$EDGES"""),
  "act": "$ACT",
  "n_freq": int($N_FREQ),
  "lr": float($LR),
  "local_epochs": int($EPOCHS),
  "n_samples": int($N_SAMPLES),
  "grid": int($GRID),
  "max_rounds": int($MAX_ROUNDS),
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
            -J "${ALG_TAG}_1n${N_CLIENTS}_g${REQUEST_GPUS}_epochs${EPOCHS}_tb${TIME_BUDGET}s"
            -p "${PARTITION}"
            -o "slurm-%A.out"
          )
          EXPORTS="ALL,N_CLIENTS=$N_CLIENTS,CFG_FILE=$CFG_FILE,REQUEST_GPUS=$REQUEST_GPUS"

          echo "[submit] #$idx  sbatch ${SBATCH_ARGS[*]}"
          sbatch "${SBATCH_ARGS[@]}" --export="$EXPORTS" weight_run_fedavg.sh

          sleep 0.2
done

echo "[done] submitted ${idx} jobs."

