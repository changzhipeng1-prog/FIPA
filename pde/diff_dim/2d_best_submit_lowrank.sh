#!/bin/bash
# 2D维度最优参数配置 - LOCAL_EPOCHS=100
# 写配置(JSON) -> 串行 sbatch 多个单节点作业 -> 作业内用 mpirun 起 N_CLIENTS 进程（PINN lowrank）
set -euo pipefail

############ 实验配置（2D最优参数） ############
LAYERS="${LAYERS:-[1,32,32,32,1]}"
EDGES="${EDGES:-[0,0.5,1.0]}"  # 2 clients for 2D
ACT="${ACT:-tanh}"
D_DIM="${D_DIM:-2}"  # 2D
LR="${LR:-1e-4}"

# 2D最优参数（固定值）
N_SAMPLES="${N_SAMPLES:-2000}"   # collocation points per client
N_BOUNDARY="${N_BOUNDARY:-200}"  # boundary points for boundary clients
LAMBDA_BC="${LAMBDA_BC:-100.0}" # boundary condition weight
GRID="${GRID:-1024}"
MAX_ROUNDS="${MAX_ROUNDS:-10000}"

# 关键超参：2D最优配置
LOCAL_EPOCHS="${LOCAL_EPOCHS:-300}"
STEP_SIZE="${STEP_SIZE:-0.3}"
TOP_K="${TOP_K:-50}"

PINV_TOL="${PINV_TOL:-1e-5}"

# 时间预算（秒）- 2D最优配置
TIME_BUDGET="${TIME_BUDGET:-60}"

# 可选
SEED="${SEED:-0}"
ALG_TAG="${ALG_TAG:-pinn_lowrank}"

# 计算 N_CLIENTS（需要在OUT_ROOT之前计算）
N_CLIENTS=$(python - <<PY
import ast
edges = ast.literal_eval("""$EDGES""")
print(len(edges)-1)
PY
)

# 输出路径包含客户端数量和lambda_bc和维度
OUT_ROOT="${OUT_ROOT:-Results_pinn_lowrank_d${D_DIM}D_N${N_CLIENTS}_bc${LAMBDA_BC}}"

############ 资源（单节点多卡） ############
PARTITION="${PARTITION:-general}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
REQUEST_GPUS="${REQUEST_GPUS:-2}"  # 2D: 2 clients, 推荐2个GPU（可根据可用资源调整）

# 固定使用的GPU池（集群GPU 3,4,5,6,7）
ALLOWED_GPUS="${ALLOWED_GPUS:-3,4,5,6,7}"

# N_CLIENTS already computed above
if [[ "$N_CLIENTS" -le 0 ]]; then
  echo "[ERR] EDGES 无效（长度<=1）" >&2; exit 1
fi

TOTAL_SAMPLES=$(( N_CLIENTS * N_SAMPLES ))
echo "[plan] PINN 2D最优配置: clients=$N_CLIENTS; request_gpus=$REQUEST_GPUS; collocation=$N_SAMPLES/client; boundary=$N_BOUNDARY; lambda_bc=$LAMBDA_BC; total_col=$TOTAL_SAMPLES; tb=${TIME_BUDGET}s"
echo "[config] LOCAL_EPOCHS=$LOCAL_EPOCHS, STEP_SIZE=$STEP_SIZE, TOP_K=$TOP_K"

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

############ 提交单个最优配置作业 ############
idx=1
STAMP="${STAMP_GLOBAL}_i${idx}"

CFG_FILE="$CFG_DIR/${ALG_TAG}_cfg_arch_${ARCH_TAG}_N${N_CLIENTS}_E${EDGE_HASH}_bc${LAMBDA_BC}_epochs${LOCAL_EPOCHS}_step${STEP_SIZE}_k${TOP_K}_rmax${MAX_ROUNDS}_tb${TIME_BUDGET}s_seed${SEED}_${STAMP}.json"

python - <<PY
import json, ast
cfg = {
  "layers": ast.literal_eval("""$LAYERS"""),
  "edges":  ast.literal_eval("""$EDGES"""),
  "act": "$ACT",
  "d_dim": int($D_DIM),
  "lr": float($LR),
  "local_epochs": int($LOCAL_EPOCHS),
  "n_samples": int($N_SAMPLES),
  "n_boundary": int($N_BOUNDARY),
  "lambda_bc": float($LAMBDA_BC),
  "grid": int($GRID),
  "max_rounds": int($MAX_ROUNDS),
  "pinv_tol": float($PINV_TOL),
  "step_size": float($STEP_SIZE),
  "top_k": int($TOP_K),
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
  -J "${ALG_TAG}_2d_best_1n${N_CLIENTS}_g${REQUEST_GPUS}_epochs${LOCAL_EPOCHS}_step${STEP_SIZE}_k${TOP_K}_bc${LAMBDA_BC}_tb${TIME_BUDGET}s"
  -p "${PARTITION}"
  -o "slurm-%A.out"
)
          # 注意：ALLOWED_GPUS包含逗号，SLURM的--export无法正确传递
          # 运行脚本会使用默认值3,4,5,6,7，所以这里可以不传递
          EXPORTS="ALL,N_CLIENTS=$N_CLIENTS,CFG_FILE=$CFG_FILE,REQUEST_GPUS=$REQUEST_GPUS"

          echo "[submit] #$idx  sbatch ${SBATCH_ARGS[*]}"
          echo "[config] 限制GPU池: $ALLOWED_GPUS"
          sbatch "${SBATCH_ARGS[@]}" --export="$EXPORTS" pinn_run_lowrank.sh

echo "[done] submitted 1 job (2D最优配置)."

