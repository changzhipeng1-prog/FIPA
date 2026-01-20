#!/bin/bash
# 写配置(JSON) -> 串行 sbatch 多个单节点作业 -> 作业内用 mpirun 起 N_CLIENTS 进程（lowrank + fixed local training）
set -euo pipefail

############ 实验配置（可用环境变量覆盖） ############
LAYERS="${LAYERS:-[1,20,20,20,1]}"
# EDGES="${EDGES:-[0,0.1,0.3,0.5,0.6,0.7,0.85,0.9,1.0]}"  # 8 clients
EDGES="${EDGES:-[0,0.5,1.0]}"  # 2 clients
# EDGES="${EDGES:-[0,0.25,0.5,0.75,1.0]}"  # 4 client
ACT="${ACT:-tanh}"
N_FREQ="${N_FREQ:-2}" # 2 4 8 for 2 4 8 clients
LR="${LR:-1e-4}"
N_SAMPLES="${N_SAMPLES:-400}"
GRID="${GRID:-1024}"
MAX_ROUNDS="${MAX_ROUNDS:-10000}"

# 关键超参：支持单值或多值（空格/逗号/括号均可）
LOCAL_STEPS_CAP="${LOCAL_STEPS_CAP:-200}"
STEP_SIZE="${STEP_SIZE:-0.5}"
EIGVAL_THRESH="${EIGVAL_THRESH:-1.0}"
TAU_COEF="${TAU_COEF:-0.1}"

PINV_TOL="${PINV_TOL:-1e-5}"

# 时间预算（秒）
TIME_BUDGET="${TIME_BUDGET:-30}"

# 可选
SEED="${SEED:-0}"
ALG_TAG="${ALG_TAG:-adaptive}"

# 计算 N_CLIENTS（需要在OUT_ROOT之前计算）
N_CLIENTS=$(python - <<PY
import ast
edges = ast.literal_eval("""$EDGES""")
print(len(edges)-1)
PY
)

# 输出路径包含客户端数量和N_FREQ
OUT_ROOT="${OUT_ROOT:-Results_adaptive_N${N_CLIENTS}_nf${N_FREQ}}"

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

CAP_LIST=$(_norm_list "$LOCAL_STEPS_CAP")
STEP_LIST=$(_norm_list "$STEP_SIZE")
THRESH_LIST=$(_norm_list "$EIGVAL_THRESH")
TAU_LIST=$(_norm_list "$TAU_COEF")

echo "[grid] LOCAL_STEPS_CAP=($CAP_LIST)"
echo "[grid] STEP_SIZE=($STEP_LIST)"
echo "[grid] EIGVAL_THRESH=($THRESH_LIST)"
echo "[grid] TAU_COEF=($TAU_LIST)"
echo "[const] PINV_TOL=$PINV_TOL"

############ 串行提交（笛卡尔积） ############
idx=0
for CAP in $CAP_LIST; do
  for STEP in $STEP_LIST; do
    for THRESH in $THRESH_LIST; do
      for TAU in $TAU_LIST; do
          idx=$((idx+1))
          STAMP="${STAMP_GLOBAL}_i${idx}"

          CFG_FILE="$CFG_DIR/${ALG_TAG}_cfg_arch_${ARCH_TAG}_N${N_CLIENTS}_E${EDGE_HASH}_nf${N_FREQ}_cap${CAP}_step${STEP}_thresh${THRESH}_tau${TAU}_rmax${MAX_ROUNDS}_tb${TIME_BUDGET}s_seed${SEED}_${STAMP}.json"

          python - <<PY
import json, ast
cfg = {
  "layers": ast.literal_eval("""$LAYERS"""),
  "edges":  ast.literal_eval("""$EDGES"""),
  "act": "$ACT",
  "n_freq": int($N_FREQ),
  "lr": float($LR),
  "local_steps_cap": int($CAP),
  "n_samples": int($N_SAMPLES),
  "grid": int($GRID),
  "max_rounds": int($MAX_ROUNDS),
  "eigval_thresh": float($THRESH),
  "pinv_tol": float($PINV_TOL),
  "step_size": float($STEP),
  "tau_coef": float($TAU),
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
            -J "${ALG_TAG}_1n${N_CLIENTS}_g${REQUEST_GPUS}_cap${CAP}_step${STEP}_thresh${THRESH}_tau${TAU}_tb${TIME_BUDGET}s"
            -p "${PARTITION}"
            -o "slurm-%A.out"
          )
          EXPORTS="ALL,N_CLIENTS=$N_CLIENTS,CFG_FILE=$CFG_FILE,REQUEST_GPUS=$REQUEST_GPUS"

          echo "[submit] #$idx  sbatch ${SBATCH_ARGS[*]}"
          sbatch "${SBATCH_ARGS[@]}" --export="$EXPORTS" weight_run_adaptive.sh

          sleep 0.2
      done
    done
    done
done

echo "[done] submitted ${idx} jobs."

