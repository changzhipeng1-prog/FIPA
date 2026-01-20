# -*- coding: utf-8 -*-
"""Distributed adaptive low-rank matrix-weighted FL with adaptive local training steps (2D version)."""

import os
import ast
import time
import json
import argparse
import pickle

import numpy as np
from mpi4py import MPI

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.numpy.linalg import lstsq
import optax
from jax import device_put, device_get, jacfwd
from jax.flatten_util import ravel_pytree

# -------------------- MPI --------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = comm.Get_size()

# -------------------- Args -------------------
def parse_layers(s: str):
    s = s.strip()
    if s.startswith('['):
        return list(map(int, ast.literal_eval(s)))
    return list(map(int, s.split(',')))

parser = argparse.ArgumentParser()
# 2D 输入 ⇒ 默认输入维度=2
parser.add_argument("--layers", type=parse_layers, default=[2, 20, 20, 1])
# 2D 目标域与均匀剖分（Kx×Ky 个 client；world 必须等于 Kx*Ky）
parser.add_argument("--domain", type=str, default="[0,1,0,1]", help="[xmin,xmax,ymin,ymax]")
parser.add_argument("--kx", type=int, default=2, help="x 方向均分块数")
parser.add_argument("--ky", type=int, default=3, help="y 方向均分块数")
parser.add_argument("--act", type=str, default="tanh", choices=["tanh","relu"])
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--local_steps_cap", type=int, default=200, help="maximum local training steps (cap for adaptive)")
parser.add_argument("--n_samples", type=int, default=400)
parser.add_argument("--grid_x", type=int, default=256, help="x 方向测试网格点数")
parser.add_argument("--grid_y", type=int, default=256, help="y 方向测试网格点数")
parser.add_argument("--max_rounds", type=int, default=10000)
parser.add_argument("--eigval_thresh", type=float, default=1.0, help="coefficient for adaptive eigenvalue threshold")
parser.add_argument("--pinv_tol", type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=0.5, help="fixed step size for server update")
parser.add_argument(
    "--tau_coef",
    type=float,
    default=1e-2,
    help="Positive coefficient in tau_i = ceil(tau_coef / (-ln(1 - lr * lambda_min_i))).",
)
parser.add_argument("--time_budget", type=float, default=50.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_root", type=str, default="Results_adaptive_2d")
args = parser.parse_args()

# -------------------- Device -----------------
DEV = next((d for d in jax.devices() if d.platform in ("gpu", "cuda")), jax.devices()[0])
if rank == 0:
    print("[MPI] world size =", world)
print(f"[rank {rank}] device:", DEV)

# -------------------- Config -----------------
LAYERS = args.layers
ACTS = {"tanh": jnp.tanh, "relu": jax.nn.relu}
act = ACTS[args.act]

LR = float(args.lr)
LOCAL_STEPS_CAP = max(1, int(args.local_steps_cap))
N_SAMPLES = int(args.n_samples)
GRID_X = int(args.grid_x)
GRID_Y = int(args.grid_y)
MAX_ROUNDS = int(args.max_rounds)
C_COEF = float(args.eigval_thresh)
PINV_TOL = float(args.pinv_tol)
STEP_SIZE = float(args.step_size)
TAU_COEF = max(1e-12, float(args.tau_coef))
TIME_BUDGET = float(args.time_budget)
SEED = int(args.seed)

XMIN, XMAX, YMIN, YMAX = list(map(float, ast.literal_eval(args.domain)))
KX, KY = int(args.kx), int(args.ky)
assert KX > 0 and KY > 0, "[Error] 2D 模式下需要 kx>0 且 ky>0"

N_CLIENTS = KX * KY
if world != N_CLIENTS:
    if rank == 0:
        raise SystemExit(f"[Error] require world == N_CLIENTS, got world={world}, N_CLIENTS={N_CLIENTS}")
    raise SystemExit

if rank == 0:
    print(f"[config] 2D domain: [{XMIN},{XMAX}]x[{YMIN},{YMAX}], KX={KX}, KY={KY}, total_clients={N_CLIENTS}")
    print(
        f"[config] adaptive low-rank: lr={LR}, local_steps_cap={LOCAL_STEPS_CAP}, step_size={STEP_SIZE}, "
        f"eigval_thresh={C_COEF}, tau_coef={TAU_COEF}, pinv_tol={PINV_TOL}"
    )

# -------------------- 输出目录 --------------------
run_name = (
    f"adaptive_2d_tb-{int(TIME_BUDGET)}s_mr-{MAX_ROUNDS}"
    f"_cap-{LOCAL_STEPS_CAP}_adapt-{C_COEF}_tau-{TAU_COEF}_step{STEP_SIZE}_K{KX}x{KY}"
)
out_dir = os.path.join(args.out_root, run_name)
os.makedirs(out_dir, exist_ok=True)

local_metrics_path = os.path.join(out_dir, f"metrics_rank{rank:03d}.csv")
if not os.path.exists(local_metrics_path):
    with open(local_metrics_path, "w") as f:
        f.write("round,g_norm,r_norm,lambda_min,local_steps\n")

# -------------------- Model ------------------
def init_params(rng, layers):
    params = []
    keys = jax.random.split(rng, len(layers) - 1)
    for k, (din, dout) in zip(keys, zip(layers[:-1], layers[1:])):
        W = jax.random.normal(k, (dout, din)) * jnp.sqrt(1.0 / din)
        b = jnp.zeros((dout,))
        params.append({"W": W, "b": b})
    return jax.tree_util.tree_map(lambda a: device_put(a, DEV), params)

def forward(params, x):
    h = x
    for i, lyr in enumerate(params):
        z = h @ lyr["W"].T + lyr["b"]
        h = z if i == len(params) - 1 else act(z)
    return h

def mse_loss(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

optimizer = optax.adam(LR)

@jax.jit
def train_step(params, opt_state, x_full, y_full):
    loss, grads = jax.value_and_grad(mse_loss)(params, x_full, y_full)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def eigs_above_thresh(JTJ, thr_rel):
    """
    Compute eigenvalues above adaptive threshold.
    Returns (λ, Q) where λ are eigenvalues above threshold and Q are corresponding eigenvectors.
    """
    S = 0.5 * (JTJ + JTJ.T)  # Ensure symmetry
    w, V = jnp.linalg.eigh(S)
    if w.size == 0:
        return w, V[:, :0]
    w_max = w[-1]
    cutoff = jnp.asarray(thr_rel, w.dtype) * w_max
    mask = w > cutoff
    lam = w[mask]
    Q = V[:, mask]
    return lam, Q

def apply_JTJ_eigs_to_vec(lam, Q, v):
    """
    Apply low-rank Hessian approximation to vector: H @ v where H = Q @ diag(λ) @ Q^T
    """
    if lam.size == 0:
        return jnp.zeros_like(v)
    alpha = Q.T @ v
    return Q @ (lam * alpha)

# -------------------- Target Function -------------------
def target_fn_2d(xy):
    """
    Evaluate the 2D target function: sum of 6 anisotropic Gaussian components
    This is a more challenging 2D test function than simple sine products.
    xy: (N, 2) array where xy[:, 0] is x-coordinate and xy[:, 1] is y-coordinate
    Returns: (N, 1) array
    """
    comps = [
        (1.40, 0.18, 0.28, 0.06, 0.05),  # (A, mx, my, sx, sy)
        (0.90, 0.50, 0.28, 0.10, 0.06),
        (0.95, 0.82, 0.28, 0.11, 0.07),
        (0.75, 0.18, 0.72, 0.07, 0.06),
        (0.70, 0.50, 0.82, 0.07, 0.07),
        (0.80, 0.82, 0.74, 0.10, 0.07),
    ]
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    z = jnp.zeros_like(x)
    for (A, mx, my, sx, sy) in comps:
        z = z + A * jnp.exp(-0.5 * (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2))
    return z

# -------------------- 2D Data：每个 rank 的本地矩形内撒点 -------------------
# 计算当前client在2D网格中的位置
ix = rank % KX
iy = rank // KX

edges_x = jnp.linspace(XMIN, XMAX, KX + 1)
edges_y = jnp.linspace(YMIN, YMAX, KY + 1)
xL, xR = float(edges_x[ix]), float(edges_x[ix+1])
yB, yT = float(edges_y[iy]), float(edges_y[iy+1])

N_LOC = N_SAMPLES
key = jax.random.PRNGKey(SEED + 12345 + rank)
u = jax.random.uniform(key, (N_LOC, 2))
x_loc = jnp.column_stack([
    xL + (xR - xL) * u[:, 0],
    yB + (yT - yB) * u[:, 1],
])
y_loc = target_fn_2d(x_loc)

xi, yi = device_put(x_loc, DEV), device_put(y_loc, DEV)

if rank == 0:
    print(f"[data] per-client sizes: {[N_LOC] * N_CLIENTS} (total={N_LOC * N_CLIENTS})")
    print(f"[data] 2D domain: [{XMIN},{XMAX}]x[{YMIN},{YMAX}], grid: {KX}x{KY}")

# Gather all training data to rank 0 for training MSE evaluation
x_train_host = np.asarray(device_get(xi))
y_train_host = np.asarray(device_get(yi))
train_data_list = comm.gather((x_train_host, y_train_host), root=0)

if rank == 0:
    # Concatenate all training data
    x_train_all = np.concatenate([item[0] for item in train_data_list], axis=0)
    y_train_all = np.concatenate([item[1] for item in train_data_list], axis=0)
    x_train_all = device_put(jnp.asarray(x_train_all), DEV)
    y_train_all = device_put(jnp.asarray(y_train_all), DEV)
else:
    x_train_all = None
    y_train_all = None

# 评估网格：2D meshgrid
xs_lin = jnp.linspace(XMIN, XMAX, GRID_X)
ys_lin = jnp.linspace(YMIN, YMAX, GRID_Y)
XX, YY = jnp.meshgrid(xs_lin, ys_lin, indexing="xy")
xs = jnp.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
ys_true = target_fn_2d(xs)

xs = device_put(xs, DEV)
ys_true = device_put(ys_true, DEV)

# -------------------- Params init + 广播 --------------------
k0 = jax.random.PRNGKey(0)
params0 = init_params(k0, LAYERS) if rank == 0 else None

def tree_to_host(t):
    return jax.tree_util.tree_map(lambda a: np.asarray(device_get(a)), t)

def tree_to_device(t):
    return jax.tree_util.tree_map(lambda a: device_put(jnp.asarray(a), DEV), t)

params_host = comm.bcast(tree_to_host(params0) if rank == 0 else None, root=0)
params = tree_to_device(params_host)

theta_ref, unravel_ref = ravel_pytree(params)
P = int(theta_ref.size)

# -------------------- 编译：theta -> (J^T J, J^T r) --------------------
EMPTY_LOCAL = (xi.shape[0] == 0)
if not EMPTY_LOCAL:
    B = int(forward(params, xi).size)

    def y_flat_ref(theta_vec):
        p = unravel_ref(theta_vec)
        return forward(p, xi).reshape(-1)

    base_jac = jacfwd(y_flat_ref) if B >= P else jax.jacrev(y_flat_ref)

    @jax.jit
    def jtj_and_g(theta_vec):
        J = base_jac(theta_vec)
        r = y_flat_ref(theta_vec) - yi.reshape(-1)
        r_norm = jnp.linalg.norm(r)
        return J.T @ J, J.T @ r, r_norm

    _H, _g, _rn = jtj_and_g(theta_ref)
    _H.block_until_ready(); _g.block_until_ready(); _rn.block_until_ready()
else:
    def jtj_and_g(theta_vec):
        return (jnp.zeros((P, P), theta_vec.dtype),
                jnp.zeros((P,), theta_vec.dtype),
                jnp.asarray(0.0, theta_vec.dtype))

if rank == 0:
    print("[jit] per-rank (JTJ, JTr) kernel compiled.")

# ===== Warm-up =====
comm.Barrier()
opt_state_warm = optimizer.init(params)
p_warm, _ = train_step(params, opt_state_warm, xi, yi)
jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_warm)
_ = mse_loss(params, xs, ys_true)
_.block_until_ready()
if not EMPTY_LOCAL:
    H0_warm, _, _ = jtj_and_g(theta_ref)
    H0_warm.block_until_ready()
    _, _ = eigs_above_thresh(H0_warm, 0.01)  # Warm-up with small threshold
    _.block_until_ready()
if rank == 0:
    H0 = jnp.zeros((P, P), dtype=theta_ref.dtype)
    b0 = jnp.zeros((P,), dtype=theta_ref.dtype)
    _ = lstsq(H0, b0, rcond=PINV_TOL)[0]
    _.block_until_ready()
comm.Barrier()

# 初始化日志
if rank == 0:
    with open(os.path.join(out_dir, "global_test.csv"), "w") as f:
        f.write("round,global_test_mse,global_train_mse,thr_rel_list,kept_min,kept_max,eigval_counts,local_steps_list\n")
    with open(os.path.join(out_dir, "timings.csv"), "w") as f:
        f.write("round,tJJ_max,tEig_max,tLocal_max,tComm_max,tRecon,tPinv,tUpdate,tBcast_max,tRound,comm_total,local_total,mwfl_total\n")
    with open(os.path.join(out_dir, "clients_metrics.csv"), "w") as f:
        f.write("round,global_test_mse,client_test_mse\n")

# -------------------- Loop --------------------
time_used_total = 0.0
lambda_min_last = None
for r in range(1, MAX_ROUNDS + 1):
    comm.Barrier()
    # ---- start round timer ----
    t_round0 = time.perf_counter()

    theta0_vec, unravel0 = ravel_pytree(params)

    # ---- time: local JTJ + g ----
    t0 = time.perf_counter()
    JTJ, g_i, r_norm_jit = jtj_and_g(theta0_vec)
    JTJ.block_until_ready(); g_i.block_until_ready(); r_norm_jit.block_until_ready()
    t_jj = time.perf_counter() - t0  # end JTJ timing

    r_norm = float(r_norm_jit)
    g_norm = float(jnp.linalg.norm(g_i))
    tComm_accum = 0.0

    thr_cap = 0.99
    # ---- time: gather local g vectors ---- (always adaptive)
    t_cg0 = time.perf_counter()
    g_host = np.asarray(device_get(g_i))
    g_list = comm.gather(g_host, root=0)
    t_comm_gather_g = time.perf_counter() - t_cg0  # end gather time
    tComm_accum += t_comm_gather_g

    if rank == 0:
        g_sum = np.sum(g_list, axis=0) if len(g_list) > 0 else None
        g_sum_norm = float(np.linalg.norm(g_sum)) if g_sum is not None else 0.0
        eps = 1e-12
        k_round = max(1, r)
        thr_arr = []
        for gi in g_list:
            gi_norm = float(np.linalg.norm(gi))
            val = C_COEF / (k_round * (gi_norm + g_sum_norm + eps))
            thr_arr.append(float(np.clip(val, 0.0, thr_cap)))
    else:
        thr_arr = None

    # ---- time: broadcast thresholds ----
    t_cb0 = time.perf_counter()
    thr_arr = comm.bcast(thr_arr, root=0)
    t_comm_bcast_thr = time.perf_counter() - t_cb0  # end broadcast time
    tComm_accum += t_comm_bcast_thr

    thr_rel_i = float(thr_arr[rank])

    # ---- time: eigen decomposition with adaptive threshold ----
    t1 = time.perf_counter()
    lam, Q = eigs_above_thresh(JTJ, thr_rel_i)
    lam.block_until_ready(); Q.block_until_ready()
    t_eig = time.perf_counter() - t1  # end eigen timing
    tEig_max = comm.allreduce(t_eig, op=MPI.MAX)
    lambda_min_local = float(jnp.min(lam)) if lam.size > 0 else float("nan")

    # ---- compute adaptive local steps ----
    lambda_min_safe = float(lambda_min_local)
    if not np.isfinite(lambda_min_safe) or lambda_min_safe <= 0.0:
        if lambda_min_last is not None:
            lambda_min_safe = lambda_min_last
        else:
            lambda_min_safe = 1e-6

    decay = 1.0 - LR * lambda_min_safe
    if not np.isfinite(decay) or not (0.0 < decay < 1.0):
        if lambda_min_last is not None:
            lambda_min_safe = lambda_min_last
            decay = 1.0 - LR * lambda_min_safe
        else:
            raise RuntimeError(
                f"Invalid decay factor {decay} computed from lr={LR} and lambda_min={lambda_min_safe}"
            )

    ln_decay = float(np.log(decay))
    if not np.isfinite(ln_decay) or ln_decay >= 0.0:
        if lambda_min_last is not None:
            lambda_min_safe = lambda_min_last
            decay = 1.0 - LR * lambda_min_safe
            ln_decay = float(np.log(decay))
        else:
            raise RuntimeError(
                f"Non-negative log decay {ln_decay} with decay={decay} at round {r} on rank {rank}"
            )

    steps_est = TAU_COEF / (-ln_decay)
    if not np.isfinite(steps_est) or steps_est <= 0.0:
        if lambda_min_last is not None:
            lambda_min_safe = lambda_min_last
            decay = 1.0 - LR * lambda_min_safe
            ln_decay = float(np.log(decay))
            steps_est = TAU_COEF / (-ln_decay)
        else:
            raise RuntimeError(
                f"Non-positive tau estimate {steps_est} derived from ln_decay={ln_decay} at round {r}"
            )

    local_steps_i = int(np.clip(int(np.ceil(steps_est)), 1, LOCAL_STEPS_CAP))
    lambda_min_last = lambda_min_safe

    # ---- time: local training (Adam) with adaptive steps ----
    t2 = time.perf_counter()
    p_i = params
    opt_state = optimizer.init(params)
    for _ in range(local_steps_i):
        p_i, opt_state = train_step(p_i, opt_state, xi, yi)
    th_i, _ = ravel_pytree(p_i)
    th_i.block_until_ready()
    t_local = time.perf_counter() - t2  # end local training time
    delta_i = th_i - theta0_vec

    tJJ_max = comm.allreduce(t_jj, op=MPI.MAX)
    tLocal_max = comm.allreduce(t_local, op=MPI.MAX)

    # ---- time: gather low-rank representation (λ, Q, d) ----
    t_c0 = time.perf_counter()
    lam_host = np.asarray(device_get(lam))
    Q_host = np.asarray(device_get(Q))
    delta_host = np.asarray(device_get(delta_i))
    payload = {"lam": lam_host, "Q": Q_host, "d": delta_host}
    gathered = comm.gather(payload, root=0)
    t_comm_payload = time.perf_counter() - t_c0  # end payload gather
    tComm_accum += t_comm_payload
    tComm_max = comm.allreduce(tComm_accum, op=MPI.MAX)

    if rank == 0:
        counts = [int(item["lam"].shape[0]) for item in gathered]
        kept_all = np.concatenate([item["lam"] for item in gathered if item["lam"].size > 0], axis=0) if any(item["lam"].size > 0 for item in gathered) else np.array([], dtype=np.float32)
        kept_min = float(kept_all.min()) if kept_all.size > 0 else float("nan")
        kept_max = float(kept_all.max()) if kept_all.size > 0 else float("nan")

        # ---- time: reconstruct low-rank Hessian and aggregate ----
        t3 = time.perf_counter()
        P32 = jnp.float32
        H_sum = jnp.zeros((P, P), dtype=P32)
        g_sum_eig = jnp.zeros((P,), dtype=P32)
        H_sum.block_until_ready(); g_sum_eig.block_until_ready()
        for item in gathered:
            lam_i = device_put(jnp.asarray(item["lam"], dtype=P32), DEV)
            Q_i = device_put(jnp.asarray(item["Q"], dtype=P32), DEV)
            d_i = device_put(jnp.asarray(item["d"], dtype=P32), DEV)
            if lam_i.size > 0:
                # Reconstruct H_i = Q_i @ diag(λ_i) @ Q_i^T
                H_i = Q_i @ (lam_i[:, None] * Q_i.T)
                H_sum = H_sum + H_i
                g_sum_eig = g_sum_eig + apply_JTJ_eigs_to_vec(lam_i, Q_i, d_i)
        H_sum.block_until_ready(); g_sum_eig.block_until_ready()
        t_recon = time.perf_counter() - t3  # end reconstruct

        # ---- time: solve linear system (equivalent to pseudo-inverse) ----
        t4 = time.perf_counter()
        delta_flat = lstsq(H_sum, g_sum_eig, rcond=PINV_TOL)[0]
        delta_flat.block_until_ready()
        t_pinv = time.perf_counter() - t4  # end solve

        # ---- time: global update with fixed step size ----
        t5 = time.perf_counter()
        theta_next = theta0_vec + STEP_SIZE * delta_flat
        theta_next.block_until_ready()
        params = unravel0(theta_next)
        t_update = time.perf_counter() - t5  # end global update
    else:
        kept_min = kept_max = float("nan")
        counts = []
        local_steps_list = None
        t_recon = t_pinv = t_update = 0.0

    client_test_mse = float(mse_loss(p_i, xs, ys_true))
    test_list = comm.gather(client_test_mse, root=0)
    
    # Gather local_steps_i from all clients
    local_steps_list = comm.gather(local_steps_i, root=0)

    # ---- time: broadcast new params ----
    t_b0 = time.perf_counter()
    params_host = comm.bcast(tree_to_host(params) if rank == 0 else None, root=0)
    params = tree_to_device(params_host)
    t_bcast = time.perf_counter() - t_b0  # end params broadcast
    tBcast_max = comm.allreduce(t_bcast, op=MPI.MAX)

    if rank == 0:
        t_round = time.perf_counter() - t_round0  # end round timer
        time_this_round = float(
            tJJ_max + tEig_max + tLocal_max + tComm_max + t_recon + t_pinv + t_update + tBcast_max
        )
        time_used_total += time_this_round
        tb_str = f"{TIME_BUDGET:.1f}s" if TIME_BUDGET > 0 else "inf"

        comm_total = tComm_max + tBcast_max
        local_total = tLocal_max
        mwfl_total = tJJ_max + tEig_max + t_recon + t_pinv + t_update

        # ---- Evaluation (not included in training time) ----
        # First evaluate on test set
        Lg = float(mse_loss(params, xs, ys_true))
        # Then evaluate on training set
        Lg_train = float(mse_loss(params, x_train_all, y_train_all))

        print(f"[round {r:03d}] test_MSE={Lg:.6e}, train_MSE={Lg_train:.6e} | "
              f"JJ(max)={tJJ_max:.3f}s, eig(max)={tEig_max:.3f}s, "
              f"comm={comm_total:.3f}s, local={local_total:.3f}s, mwfl={mwfl_total:.3f}s, total={t_round:.3f}s | "
              f"used={time_used_total:.1f}/{tb_str}")

        # Save both test and train MSE to CSV
        with open(os.path.join(out_dir, "global_test.csv"), "a") as f:
            f.write(f"{r},{Lg:.10e},{Lg_train:.10e},\"{json.dumps(thr_arr)}\",{kept_min:.6e},{kept_max:.6e},\"{json.dumps(counts)}\",\"{json.dumps(local_steps_list)}\"\n")
        with open(os.path.join(out_dir, "timings.csv"), "a") as f:
            f.write(
                f"{r},{tJJ_max:.6f},{tEig_max:.6f},{tLocal_max:.6f},{tComm_max:.6f},"
                f"{t_recon:.6f},{t_pinv:.6f},{t_update:.6f},{tBcast_max:.6f},{t_round:.6f},"
                f"{comm_total:.6f},{local_total:.6f},{mwfl_total:.6f}\n"
            )
        with open(os.path.join(out_dir, "clients_metrics.csv"), "a") as f:
            f.write(f"{r},{Lg:.10e},\"{json.dumps(list(test_list))}\"\n")
    else:
        tEig_max = 0.0

    with open(local_metrics_path, "a") as f:
        f.write(f"{r},{g_norm:.10e},{r_norm:.10e},{lambda_min_local:.10e},{local_steps_i}\n")

    if rank == 0:
        stop_flag_local = True if (TIME_BUDGET > 0 and time_used_total >= TIME_BUDGET) else False
        if stop_flag_local:
            print(f"[STOP] reach time budget after round {r}: used={time_used_total:.1f}s / {TIME_BUDGET:.1f}s")
    else:
        stop_flag_local = False
    stop_flag = comm.bcast(stop_flag_local, root=0)
    if stop_flag:
        break

comm.Barrier()

if rank == 0:
    y_pred = forward(params, xs)
    y_pred.block_until_ready()
    x_host = np.asarray(device_get(xs))  # (N, 2) for 2D
    y_true_host = np.asarray(device_get(ys_true)).reshape(-1)
    y_pred_host = np.asarray(device_get(y_pred)).reshape(-1)
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w") as f:
        f.write("x,y,y_true,y_pred\n")
        for i in range(len(y_true_host)):
            f.write(f"{x_host[i, 0]:.10e},{x_host[i, 1]:.10e},{y_true_host[i]:.10e},{y_pred_host[i]:.10e}\n")

    # Save final model parameters
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.asarray(device_get(x))
        return x
    
    params_numpy = jax.tree_util.tree_map(to_numpy, params)
    params_file = os.path.join(out_dir, "final_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params_numpy, f)
    
    print("Saved logs to:")
    print(" -", os.path.join(out_dir, "global_test.csv"))
    print(" -", os.path.join(out_dir, "timings.csv"))
    print(" -", os.path.join(out_dir, "predictions.csv"))
    print(" -", params_file)

