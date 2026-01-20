# -*- coding: utf-8 -*-
"""Distributed FedAvg PINN for PDE with boundary conditions."""

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
parser.add_argument("--layers", type=parse_layers, default=[1, 20, 20, 1])
parser.add_argument("--edges", type=str, default="[-1.0,-0.5,0.0,0.5,1.0]")
parser.add_argument("--act", type=str, default="tanh", choices=["tanh","relu"])
parser.add_argument("--d_dim", type=int, default=1, help="spatial dimension (1D, 2D, 3D, etc.)")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--local_epochs", type=int, default=300)
parser.add_argument("--n_samples", type=int, default=400, help="number of collocation points per client")
parser.add_argument("--n_boundary", type=int, default=20, help="number of boundary points per client")
parser.add_argument("--grid", type=int, default=1024)
parser.add_argument("--max_rounds", type=int, default=10000)
parser.add_argument("--lambda_bc", type=float, default=100.0, help="boundary condition weight")
parser.add_argument("--time_budget", type=float, default=50.0)
parser.add_argument("--out_root", type=str, default="Results_pinn_fedavg")
args = parser.parse_args()

# -------------------- Device -----------------
DEV = next((d for d in jax.devices() if d.platform in ("gpu", "cuda")), jax.devices()[0])
if rank == 0:
    print("[MPI] world size =", world)
print(f"[rank {rank}] device:", DEV)

# -------------------- Config -----------------
D_DIM = max(1, int(args.d_dim))  # Spatial dimension
EDGES = list(map(float, ast.literal_eval(args.edges)))
ACTS = {"tanh": jnp.tanh, "relu": jax.nn.relu}
act = ACTS[args.act]

# Adjust input layer dimension for d-dimensional case
# Model takes d-dim input, outputs 1-dim (scalar)
LAYERS = list(args.layers)  # Create a copy
if LAYERS[0] == 1:
    LAYERS[0] = D_DIM

LR = float(args.lr)
LOCAL_EPOCHS = max(1, int(args.local_epochs))
N_SAMPLES = int(args.n_samples)
N_BOUNDARY = int(args.n_boundary)
GRID = int(args.grid)
MAX_ROUNDS = int(args.max_rounds)
LAMBDA_BC = float(args.lambda_bc)
TIME_BUDGET = float(args.time_budget)

N_CLIENTS = len(EDGES) - 1
if world != N_CLIENTS:
    if rank == 0:
        raise SystemExit(f"[Error] require world == N_CLIENTS, got world={world}, N_CLIENTS={N_CLIENTS}")
    raise SystemExit

if rank == 0:
    print(f"[config] PINN FedAvg: d_dim={D_DIM}, lr={LR}, local_epochs={LOCAL_EPOCHS}")
    print(f"[config] lambda_bc={LAMBDA_BC}, n_boundary={N_BOUNDARY}, layers={LAYERS}")

# -------------------- 输出目录 --------------------
run_name = (
    f"pinn_fedavg_d{D_DIM}D_tb-{int(TIME_BUDGET)}s_mr-{MAX_ROUNDS}"
    f"_epochs{LOCAL_EPOCHS}_bc{LAMBDA_BC}"
)
out_dir = os.path.join(args.out_root, run_name)
os.makedirs(out_dir, exist_ok=True)

local_metrics_path = os.path.join(out_dir, f"metrics_rank{rank:03d}.csv")
if not os.path.exists(local_metrics_path):
    with open(local_metrics_path, "w") as f:
        f.write("round,pde_norm,bc_norm,total_norm,local_epochs\n")

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

# -------------------- PDE Definition --------------------
# For d-dimensional case: u(x_1, ..., x_d) = sum_{i=1}^d u_1d(x_i)
# PDE: -∇²u = -sum_{i=1}^d ∂²u/∂x_i² = f(x_1, ..., x_d) = sum_{i=1}^d f_1d(x_i)
# Here u_1d(x) = sin(πx), f_1d(x) = π² sin(πx)

def source_term_1d(x):
    """1D source term: f_1d(x) = π² sin(πx)"""
    return jnp.pi**2 * jnp.sin(jnp.pi * x)

def exact_solution_1d(x):
    """1D exact solution: u_1d(x) = sin(πx)"""
    return jnp.sin(jnp.pi * x)

def source_term(x):
    """d-dimensional source term: f(x_1, ..., x_d) = sum_{i=1}^d f_1d(x_i)"""
    # x shape: (..., d_dim)
    return jnp.sum(source_term_1d(x), axis=-1)

def exact_solution(x):
    """d-dimensional exact solution: u(x_1, ..., x_d) = sum_{i=1}^d u_1d(x_i)"""
    # x shape: (..., d_dim)
    return jnp.sum(exact_solution_1d(x), axis=-1)

def boundary_value(x):
    """Dirichlet boundary condition: u(boundary) = exact_solution(x)"""
    return exact_solution(x)

# -------------------- PINN Loss --------------------
def pde_residual(params, x_col):
    """
    Compute PDE residual: -∇²u - f(x) for d-dimensional case
    where -∇²u = -sum_{i=1}^d ∂²u/∂x_i²
    Returns residual vector of shape (n_col,)
    """
    def u_fn(x):
        """u(x) where x is a d-dim vector"""
        return forward(params, x.reshape(1, -1)).squeeze()
    
    def laplacian_u(x):
        """Compute -∇²u = -sum_{i=1}^d ∂²u/∂x_i² using Hessian"""
        # Compute full Hessian matrix (d x d)
        H = jax.hessian(u_fn)(x)
        # Extract diagonal (second derivatives w.r.t. each coordinate)
        diag = jnp.diag(H)  # shape: (d,)
        # Laplacian is sum of diagonal elements
        return -jnp.sum(diag)
    
    # Compute Laplacian for each collocation point
    lap_u = jax.vmap(laplacian_u)(x_col)  # shape: (n_col,)
    f_val = source_term(x_col)  # shape: (n_col,)
    residual = lap_u - f_val
    return residual

def boundary_residual(params, x_bc):
    """
    Compute boundary residual: u(x_bc) - g(x_bc)
    Returns residual vector of shape (n_bc,)
    """
    u_val = forward(params, x_bc).squeeze()
    g_val = boundary_value(x_bc).squeeze()
    residual = u_val - g_val
    return residual

def pinn_loss_components(params, x_col, x_bc):
    """
    Compute PDE and BC residuals separately for monitoring.
    Returns (loss_pde, loss_bc, loss_total)
    """
    r_pde = pde_residual(params, x_col)
    r_bc = boundary_residual(params, x_bc)
    loss_pde = jnp.mean(r_pde ** 2)
    loss_bc = jnp.mean(r_bc ** 2)
    loss_total = loss_pde + LAMBDA_BC * loss_bc
    return loss_total, loss_pde, loss_bc

def pinn_loss(params, x_col, x_bc):
    """Total PINN loss for optimizer"""
    loss_total, _, _ = pinn_loss_components(params, x_col, x_bc)
    return loss_total

optimizer = optax.adam(LR)

@jax.jit
def train_step(params, opt_state, x_col, x_bc):
    loss, grads = jax.value_and_grad(pinn_loss)(params, x_col, x_bc)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# -------------------- Data (collocation + boundary) -------------------
# Domain partition strategy:
# - EDGES defines partition of the FIRST dimension: [x_min, edge1, edge2, ..., x_max]
# - For d-dimensional domain: each client owns a "strip" region
#   * Client i: x_1 in [EDGES[i], EDGES[i+1]], x_2,...,x_d in [x_min, x_max]
edges_np = np.asarray(EDGES, dtype=np.float64)
CID = rank
L, R = float(edges_np[CID]), float(edges_np[CID + 1])
x_min, x_max = float(EDGES[0]), float(EDGES[-1])

# Collocation points (interior) - d-dimensional
key = jax.random.PRNGKey(12345 + CID)
if D_DIM == 1:
    # 1D case: only first dimension
    key, subkey = jax.random.split(key)
    x_col = jax.random.uniform(subkey, (N_SAMPLES, 1), minval=L, maxval=R)
else:
    # d-dim case: first dimension in [L,R], others in [x_min, x_max]
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    x_col_dim1 = jax.random.uniform(subkey1, (N_SAMPLES, 1), minval=L, maxval=R)
    x_col_rest = jax.random.uniform(subkey2, (N_SAMPLES, D_DIM - 1), minval=x_min, maxval=x_max)
    x_col = jnp.concatenate([x_col_dim1, x_col_rest], axis=1)

# Boundary points: d-dimensional boundaries
key_bc = key
x_bc_list = []

# Boundary 1: Left boundary of first dimension (x_1 = x_min) - only for Client 0
if L == x_min:
    key_bc, subkey = jax.random.split(key_bc)
    x_bc_dim1_left = jax.random.uniform(subkey, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_dim1_left = x_bc_dim1_left.at[:, 0].set(x_min)
    x_bc_list.append(x_bc_dim1_left)

# Boundary 2: Right boundary of first dimension (x_1 = x_max) - only for last client
if R == x_max:
    key_bc, subkey = jax.random.split(key_bc)
    x_bc_dim1_right = jax.random.uniform(subkey, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_dim1_right = x_bc_dim1_right.at[:, 0].set(x_max)
    x_bc_list.append(x_bc_dim1_right)

# Boundaries 3-2*d: Other dimensions (x_2, x_3, ..., x_d)
for dim_idx in range(1, D_DIM):
    # Left boundary: x_{dim_idx+1} = x_min
    key_bc, subkey1 = jax.random.split(key_bc)
    key_bc, subkey2 = jax.random.split(key_bc)
    x_bc_other_left = jax.random.uniform(subkey1, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_other_left = x_bc_other_left.at[:, 0].set(
        jax.random.uniform(subkey2, (N_BOUNDARY,), minval=L, maxval=R)
    )
    x_bc_other_left = x_bc_other_left.at[:, dim_idx].set(x_min)
    x_bc_list.append(x_bc_other_left)
    
    # Right boundary: x_{dim_idx+1} = x_max
    key_bc, subkey1 = jax.random.split(key_bc)
    key_bc, subkey2 = jax.random.split(key_bc)
    x_bc_other_right = jax.random.uniform(subkey1, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_other_right = x_bc_other_right.at[:, 0].set(
        jax.random.uniform(subkey2, (N_BOUNDARY,), minval=L, maxval=R)
    )
    x_bc_other_right = x_bc_other_right.at[:, dim_idx].set(x_max)
    x_bc_list.append(x_bc_other_right)

# Concatenate all boundary points
if len(x_bc_list) > 0:
    x_bc = jnp.concatenate(x_bc_list, axis=0)
else:
    x_bc = jnp.zeros((0, D_DIM), dtype=jnp.float64)

x_col_dev = device_put(x_col, DEV)
x_bc_dev = device_put(x_bc, DEV)

n_bc_local = x_bc.shape[0]
n_bc_total = comm.allreduce(n_bc_local, op=MPI.SUM)
n_bc_first_dim = (1 if L == x_min else 0) + (1 if R == x_max else 0)
n_bc_other_dims = 2 * (D_DIM - 1)
n_bc_expected = (n_bc_first_dim + n_bc_other_dims) * N_BOUNDARY

# Calculate data size for FedAvg weighting
n_data_local = x_col_dev.shape[0] + x_bc_dev.shape[0]  # Total data points for this client

if rank == 0:
    print(f"[data] d_dim={D_DIM}, N_CLIENTS={N_CLIENTS}")
    print(f"[data] domain partition (first dim): {EDGES}")
    print(f"[data] per-client collocation: {N_SAMPLES} (in strip: x_1 in [L,R], x_2..x_{D_DIM} in [{x_min},{x_max}])")
    print(f"[data] boundary strategy: each client handles boundaries intersecting its region")
    print(f"[data]   - First dim boundaries: {n_bc_first_dim} face(s) per boundary client")
    print(f"[data]   - Other dim boundaries: {n_bc_other_dims} faces (2 per dim for x_2..x_{D_DIM})")
    print(f"[data]   - Expected boundary points per client: {n_bc_expected}")
    print(f"[data] boundary points: total={n_bc_total} across all clients")
    print(f"[data] domain: [{EDGES[0]}, {EDGES[-1]}]^{D_DIM} (d={D_DIM})")

# Test points for evaluation - d-dimensional (random sampling)
x_min, x_max = float(EDGES[0]), float(EDGES[-1])
key_test = jax.random.PRNGKey(99999)
key_test, subkey_test = jax.random.split(key_test)
xs = jax.random.uniform(subkey_test, (GRID, D_DIM), minval=x_min, maxval=x_max)
xs = device_put(xs, DEV)
ys_true = device_put(exact_solution(xs), DEV)

# -------------------- Params init + broadcast --------------------
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
p_warm, _ = train_step(params, opt_state_warm, x_col_dev, x_bc_dev)
jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_warm)
_ = forward(params, xs)
_.block_until_ready()
comm.Barrier()

# 初始化日志
if rank == 0:
    with open(os.path.join(out_dir, "global_test.csv"), "w") as f:
        f.write("round,rel_l2_error\n")
    with open(os.path.join(out_dir, "timings.csv"), "w") as f:
        f.write("round,tLocal_max,tComm_max,tAggr,tBcast_max,tRound,comm_total,local_total\n")

# -------------------- Training Loop (FedAvg) --------------------
time_used_total = 0.0
for r in range(1, MAX_ROUNDS + 1):
    comm.Barrier()
    t_round0 = time.perf_counter()
    
    # ---- Local training (Adam) ----
    t2 = time.perf_counter()
    p_i = params
    opt_state = optimizer.init(params)
    for _ in range(LOCAL_EPOCHS):
        p_i, opt_state = train_step(p_i, opt_state, x_col_dev, x_bc_dev)
    p_i_blocked = jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_i)
    t_local = time.perf_counter() - t2
    tLocal_max = comm.allreduce(t_local, op=MPI.MAX)
    
    # ---- Compute loss components for monitoring ----
    _, r_pde_norm, r_bc_norm = pinn_loss_components(p_i, x_col_dev, x_bc_dev)
    r_total_norm = jnp.sqrt(jnp.mean((pde_residual(p_i, x_col_dev) ** 2)) + 
                           LAMBDA_BC * jnp.mean((boundary_residual(p_i, x_bc_dev) ** 2)))
    
    # ---- Gather local parameters and data sizes for FedAvg ----
    t_c0 = time.perf_counter()
    params_i_host = tree_to_host(p_i)
    payload = {"params": params_i_host, "n_data": n_data_local}
    gathered = comm.gather(payload, root=0)
    t_comm = time.perf_counter() - t_c0
    tComm_max = comm.allreduce(t_comm, op=MPI.MAX)
    
    if rank == 0:
        # ---- FedAvg aggregation: weighted average by data size ----
        t3 = time.perf_counter()
        # Calculate total data size
        total_data = sum(item["n_data"] for item in gathered)
        
        # Weighted average: params_global = Σ (n_i / n_total) * params_i
        params_sum = None
        for item in gathered:
            params_i = tree_to_device(item["params"])
            weight = item["n_data"] / total_data
            
            if params_sum is None:
                params_sum = jax.tree_util.tree_map(lambda p: weight * p, params_i)
            else:
                params_sum = jax.tree_util.tree_map(
                    lambda s, p: s + weight * p, params_sum, params_i
                )
        
        params = params_sum
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
        t_aggr = time.perf_counter() - t3
    else:
        t_aggr = 0.0
    
    # ---- Broadcast aggregated params ----
    t_b0 = time.perf_counter()
    params_host = comm.bcast(tree_to_host(params) if rank == 0 else None, root=0)
    params = tree_to_device(params_host)
    t_bcast = time.perf_counter() - t_b0
    tBcast_max = comm.allreduce(t_bcast, op=MPI.MAX)
    
    if rank == 0:
        t_round = time.perf_counter() - t_round0
        time_this_round = float(tLocal_max + tComm_max + t_aggr + tBcast_max)
        time_used_total += time_this_round
        
        comm_total = tComm_max + tBcast_max
        local_total = tLocal_max
        
        # ---- Evaluation (Relative L2 error on test grid) ----
        u_pred = forward(params, xs).squeeze()
        u_true = ys_true.squeeze()
        l2_norm_error = jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))
        l2_norm_true = jnp.sqrt(jnp.mean(u_true ** 2))
        rel_l2_error = float(l2_norm_error / l2_norm_true)
        
        print(f"[round {r:03d}] rel_L2_error={rel_l2_error:.6e} | "
              f"comm={comm_total:.3f}s, local={local_total:.3f}s, aggr={t_aggr:.3f}s | "
              f"used={time_used_total:.1f}/{TIME_BUDGET:.1f}s")
        
        with open(os.path.join(out_dir, "global_test.csv"), "a") as f:
            f.write(f"{r},{rel_l2_error:.10e}\n")
        with open(os.path.join(out_dir, "timings.csv"), "a") as f:
            f.write(
                f"{r},{tLocal_max:.6f},{tComm_max:.6f},{t_aggr:.6f},{tBcast_max:.6f},{t_round:.6f},"
                f"{comm_total:.6f},{local_total:.6f}\n"
            )
    
    # Save local metrics
    with open(local_metrics_path, "a") as f:
        f.write(f"{r},{float(r_pde_norm):.10e},{float(r_bc_norm):.10e},"
                f"{float(r_total_norm):.10e},{LOCAL_EPOCHS}\n")
    
    # Time budget check
    if rank == 0:
        stop_flag_local = (TIME_BUDGET > 0 and time_used_total >= TIME_BUDGET)
        if stop_flag_local:
            print(f"[STOP] reach time budget after round {r}")
    else:
        stop_flag_local = False
    stop_flag = comm.bcast(stop_flag_local, root=0)
    if stop_flag:
        break

comm.Barrier()

# -------------------- Save Results --------------------
if rank == 0:
    y_pred = forward(params, xs)
    y_pred.block_until_ready()
    x_host = np.asarray(device_get(xs))
    y_true_host = np.asarray(device_get(ys_true)).reshape(-1)
    y_pred_host = np.asarray(device_get(y_pred)).reshape(-1)
    
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w") as f:
        header = ",".join([f"x{i+1}" for i in range(D_DIM)]) + ",u_exact,u_pred\n"
        f.write(header)
        for i in range(x_host.shape[0]):
            x_row = x_host[i]
            x_str = ",".join(f"{xi:.10e}" for xi in x_row)
            f.write(f"{x_str},{y_true_host[i]:.10e},{y_pred_host[i]:.10e}\n")
    
    # Save final model parameters
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.asarray(device_get(x))
        return x
    
    params_numpy = jax.tree_util.tree_map(to_numpy, params)
    params_file = os.path.join(out_dir, "final_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params_numpy, f)
    
    print("\n✓ PINN FedAvg training completed!")
    print("Saved logs to:")
    print(" -", os.path.join(out_dir, "global_test.csv"))
    print(" -", os.path.join(out_dir, "timings.csv"))
    print(" -", pred_path)
    print(" -", params_file)

