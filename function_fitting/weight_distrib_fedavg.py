# -*- coding: utf-8 -*-
"""Distributed FedAvg FL with fixed local training steps."""

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
import optax
from jax import device_put, device_get

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
parser.add_argument("--layers", type=parse_layers, default=[1, 20, 20, 1])
parser.add_argument("--edges", type=str, default="[-1.0,-0.5,0.0,0.5,1.0]")
parser.add_argument("--act", type=str, default="tanh", choices=["tanh","relu"])
parser.add_argument("--n_freq", type=int, default=1, help="frequency parameter for sin(n_freq * π * x)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--local_epochs", type=int, default=300, help="fixed local training epochs")
parser.add_argument("--n_samples", type=int, default=400)
parser.add_argument("--grid", type=int, default=1024)
parser.add_argument("--max_rounds", type=int, default=10000)
parser.add_argument("--time_budget", type=float, default=50.0)
parser.add_argument("--out_root", type=str, default="Results_fedavg")
args = parser.parse_args()

# -------------------- Device -----------------
DEV = next((d for d in jax.devices() if d.platform in ("gpu", "cuda")), jax.devices()[0])
if rank == 0:
    print("[MPI] world size =", world)
print(f"[rank {rank}] device:", DEV)

# -------------------- Config -----------------
LAYERS = args.layers
EDGES = list(map(float, ast.literal_eval(args.edges)))
ACTS = {"tanh": jnp.tanh, "relu": jax.nn.relu}
act = ACTS[args.act]

LR = float(args.lr)
LOCAL_EPOCHS = max(1, int(args.local_epochs))
N_SAMPLES = int(args.n_samples)
GRID = int(args.grid)
MAX_ROUNDS = int(args.max_rounds)
TIME_BUDGET = float(args.time_budget)
N_FREQ = int(args.n_freq)

N_CLIENTS = len(EDGES) - 1
if world != N_CLIENTS:
    if rank == 0:
        raise SystemExit(f"[Error] require world == N_CLIENTS, got world={world}, N_CLIENTS={N_CLIENTS}")
    raise SystemExit

if rank == 0:
    print(f"[config] FedAvg: lr={LR}, local_epochs={LOCAL_EPOCHS}")

# -------------------- 输出目录 --------------------
run_name = (
    f"fedavg_tb-{int(TIME_BUDGET)}s_mr-{MAX_ROUNDS}"
    f"_epochs{LOCAL_EPOCHS}"
)
out_dir = os.path.join(args.out_root, run_name)
os.makedirs(out_dir, exist_ok=True)

local_metrics_path = os.path.join(out_dir, f"metrics_rank{rank:03d}.csv")
if not os.path.exists(local_metrics_path):
    with open(local_metrics_path, "w") as f:
        f.write("round,local_epochs\n")

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

# -------------------- Target Function -------------------
def target_fn(x, n_freq):
    """Evaluate the 1D target function: f(x) = sin(n_freq * π * x)"""
    return jnp.sin(n_freq * jnp.pi * x)

# -------------------- Data -------------------
edges_np = np.asarray(EDGES, dtype=np.float64)
CID = rank
L, R = float(edges_np[CID]), float(edges_np[CID + 1])

N_LOC = N_SAMPLES
key = jax.random.PRNGKey(12345 + CID)
x_loc = jax.random.uniform(key, (N_LOC, 1), minval=L, maxval=R)
y_loc = target_fn(x_loc, N_FREQ)

xi, yi = device_put(x_loc, DEV), device_put(y_loc, DEV)

if rank == 0:
    print(f"[data] per-client sizes: {[N_LOC] * N_CLIENTS} (total={N_LOC * N_CLIENTS})")
    print(f"[data] domain: [{EDGES[0]}, {EDGES[-1]}]")

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

# Test grid covers the full domain
x_min, x_max = float(EDGES[0]), float(EDGES[-1])
xs = device_put(jnp.linspace(x_min, x_max, GRID).reshape(-1, 1), DEV)
ys_true = device_put(target_fn(xs, N_FREQ), DEV)

# -------------------- Params init + 广播 --------------------
k0 = jax.random.PRNGKey(0)
params0 = init_params(k0, LAYERS) if rank == 0 else None

def tree_to_host(t):
    return jax.tree_util.tree_map(lambda a: np.asarray(device_get(a)), t)

def tree_to_device(t):
    return jax.tree_util.tree_map(lambda a: device_put(jnp.asarray(a), DEV), t)

params_host = comm.bcast(tree_to_host(params0) if rank == 0 else None, root=0)
params = tree_to_device(params_host)

# ===== Warm-up =====
comm.Barrier()
opt_state_warm = optimizer.init(params)
p_warm, _ = train_step(params, opt_state_warm, xi, yi)
jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_warm)
_ = mse_loss(params, xs, ys_true)
_.block_until_ready()
comm.Barrier()

# -------------------- FedAvg: compute weights based on data size --------------------
n_i = N_SAMPLES  # Each client has the same number of samples
n_total = N_CLIENTS * N_SAMPLES
weight_i = n_i / n_total  # Weight for each client (all clients have same weight if same data size)
if rank == 0:
    print(f"[FedAvg] client weights: {weight_i:.6f} (all clients have same data size)")

# 初始化日志
if rank == 0:
    with open(os.path.join(out_dir, "global_test.csv"), "w") as f:
        f.write("round,global_test_mse,global_train_mse\n")
    with open(os.path.join(out_dir, "timings.csv"), "w") as f:
        f.write("round,tLocal_max,tComm_max,tAggregate,tBcast_max,tRound,comm_total,local_total\n")
    with open(os.path.join(out_dir, "clients_metrics.csv"), "w") as f:
        f.write("round,global_test_mse,client_test_mse\n")

# -------------------- FedAvg: weighted average function --------------------
def tree_weighted_average(params_list, weights):
    """
    params_list: list of pytrees (same structure)
    weights: (m,) sum to 1
    return: weighted average pytree
    """
    def combine(*xs):
        stacked = jnp.stack(xs, axis=0)  # (m, ...)
        w = weights.reshape((weights.shape[0],) + (1,) * (stacked.ndim - 1))
        return jnp.sum(w * stacked, axis=0)
    return jax.tree_util.tree_map(combine, *params_list)

# -------------------- Loop --------------------
time_used_total = 0.0
for r in range(1, MAX_ROUNDS + 1):
    comm.Barrier()
    # ---- start round timer ----
    t_round0 = time.perf_counter()

    # ---- time: local training (Adam) with fixed epochs ----
    t2 = time.perf_counter()
    p_i = params
    opt_state = optimizer.init(params)
    for _ in range(LOCAL_EPOCHS):
        p_i, opt_state = train_step(p_i, opt_state, xi, yi)
    jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_i)
    t_local = time.perf_counter() - t2  # end local training time

    tLocal_max = comm.allreduce(t_local, op=MPI.MAX)

    # ---- time: gather local model parameters ----
    t_c0 = time.perf_counter()
    params_i_host = tree_to_host(p_i)
    gathered_params = comm.gather(params_i_host, root=0)
    t_comm_gather = time.perf_counter() - t_c0  # end gather
    tComm_accum = t_comm_gather
    tComm_max = comm.allreduce(tComm_accum, op=MPI.MAX)

    if rank == 0:
        # ---- time: FedAvg aggregation (weighted average) ----
        t3 = time.perf_counter()
        # Convert to device
        params_list = [tree_to_device(p) for p in gathered_params]
        # All clients have same data size, so equal weights
        weights = jnp.ones(N_CLIENTS) / N_CLIENTS
        params = tree_weighted_average(params_list, weights)
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
        t_aggregate = time.perf_counter() - t3  # end aggregate
    else:
        t_aggregate = 0.0

    client_test_mse = float(mse_loss(p_i, xs, ys_true))
    test_list = comm.gather(client_test_mse, root=0)

    # ---- time: broadcast new params ----
    t_b0 = time.perf_counter()
    params_host = comm.bcast(tree_to_host(params) if rank == 0 else None, root=0)
    params = tree_to_device(params_host)
    t_bcast = time.perf_counter() - t_b0  # end params broadcast
    tBcast_max = comm.allreduce(t_bcast, op=MPI.MAX)

    if rank == 0:
        t_round = time.perf_counter() - t_round0  # end round timer
        time_this_round = float(
            tLocal_max + tComm_max + t_aggregate + tBcast_max
        )
        time_used_total += time_this_round
        tb_str = f"{TIME_BUDGET:.1f}s" if TIME_BUDGET > 0 else "inf"

        comm_total = tComm_max + tBcast_max
        local_total = tLocal_max

        # ---- Evaluation (not included in training time) ----
        # First evaluate on test set
        Lg = float(mse_loss(params, xs, ys_true))
        # Then evaluate on training set
        Lg_train = float(mse_loss(params, x_train_all, y_train_all))

        print(f"[round {r:03d}] test_MSE={Lg:.6e}, train_MSE={Lg_train:.6e} | "
              f"comm={comm_total:.3f}s, local={local_total:.3f}s, total={t_round:.3f}s | "
              f"used={time_used_total:.1f}/{tb_str}")

        # Save both test and train MSE to CSV
        with open(os.path.join(out_dir, "global_test.csv"), "a") as f:
            f.write(f"{r},{Lg:.10e},{Lg_train:.10e}\n")
        with open(os.path.join(out_dir, "timings.csv"), "a") as f:
            f.write(
                f"{r},{tLocal_max:.6f},{tComm_max:.6f},"
                f"{t_aggregate:.6f},{tBcast_max:.6f},{t_round:.6f},"
                f"{comm_total:.6f},{local_total:.6f}\n"
            )
        with open(os.path.join(out_dir, "clients_metrics.csv"), "a") as f:
            f.write(f"{r},{Lg:.10e},\"{json.dumps(list(test_list))}\"\n")

    with open(local_metrics_path, "a") as f:
        f.write(f"{r},{LOCAL_EPOCHS}\n")

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
    x_host = np.asarray(device_get(xs)).reshape(-1)
    y_true_host = np.asarray(device_get(ys_true)).reshape(-1)
    y_pred_host = np.asarray(device_get(y_pred)).reshape(-1)
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w") as f:
        f.write("x,y_true,y_pred\n")
        for a, b, c in zip(x_host, y_true_host, y_pred_host):
            f.write(f"{a:.10e},{b:.10e},{c:.10e}\n")

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

