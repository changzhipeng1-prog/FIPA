# -*- coding: utf-8 -*-
"""Distributed low-rank PINN for 1D Fisher equation (KPP equation) with Dirichlet boundary conditions.
PDE: -d²u/dx² + u(1-u) = f(x),  x ∈ [0, 1]
Exact solution: u(x) = x²(1-x)  (on [0,1])
Source term: f(x) = -2 + 6x + x² - x³ - x⁴ + 2x⁵ - x⁶
"""

import os
import ast
import time
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
parser.add_argument("--layers", type=parse_layers, default=[1, 20, 20, 1])
parser.add_argument("--edges", type=str, default="[-1.0,-0.5,0.0,0.5,1.0]")
parser.add_argument("--act", type=str, default="tanh", choices=["tanh","relu"])
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--local_epochs", type=int, default=300)
parser.add_argument("--n_samples", type=int, default=400, help="number of collocation points per client")
parser.add_argument("--n_boundary", type=int, default=20, help="number of boundary points per client")
parser.add_argument("--grid", type=int, default=1024)
parser.add_argument("--max_rounds", type=int, default=10000)
parser.add_argument("--pinv_tol", type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=0.5)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--lambda_bc", type=float, default=100.0, help="boundary condition weight")
parser.add_argument("--time_budget", type=float, default=50.0)
parser.add_argument("--out_root", type=str, default="Results_pinn_lowrank")
args = parser.parse_args()

# -------------------- Device -----------------
DEV = next((d for d in jax.devices() if d.platform in ("gpu", "cuda")), jax.devices()[0])
if rank == 0:
    print("[MPI] world size =", world)
print(f"[rank {rank}] device:", DEV)

# -------------------- Config -----------------
EDGES = list(map(float, ast.literal_eval(args.edges)))
ACTS = {"tanh": jnp.tanh, "relu": jax.nn.relu}
act = ACTS[args.act]

LAYERS = list(args.layers)  # Model: input=1, output=1 (1D problem)
if LAYERS[0] != 1:
    if rank == 0:
        print(f"[Warning] First layer should be 1 for 1D problem, got {LAYERS[0]}, setting to 1")
    LAYERS[0] = 1

LR = float(args.lr)
LOCAL_EPOCHS = max(1, int(args.local_epochs))
N_SAMPLES = int(args.n_samples)
N_BOUNDARY = int(args.n_boundary)
GRID = int(args.grid)
MAX_ROUNDS = int(args.max_rounds)
PINV_TOL = float(args.pinv_tol)
STEP_SIZE = float(args.step_size)
TOP_K = max(1, int(args.top_k))
LAMBDA_BC = float(args.lambda_bc)
TIME_BUDGET = float(args.time_budget)

N_CLIENTS = len(EDGES) - 1
if world != N_CLIENTS:
    if rank == 0:
        raise SystemExit(f"[Error] require world == N_CLIENTS, got world={world}, N_CLIENTS={N_CLIENTS}")
    raise SystemExit

if rank == 0:
    print(f"[config] PINN low-rank (1D): lr={LR}, local_epochs={LOCAL_EPOCHS}, step_size={STEP_SIZE}")
    print(f"[config] top_k={TOP_K}, lambda_bc={LAMBDA_BC}, n_boundary={N_BOUNDARY}, layers={LAYERS}")

# -------------------- 输出目录 --------------------
run_name = (
    f"fisher_1D_tb-{int(TIME_BUDGET)}s_mr-{MAX_ROUNDS}"
    f"_epochs{LOCAL_EPOCHS}_step{STEP_SIZE}_k{TOP_K}_bc{LAMBDA_BC}"
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
# 1D Fisher equation: -d²u/dx² + u(1-u) = f(x)
# Exact solution: u(x) = x²(1-x)  (on [0,1])
# Source term: f(x) = -2 + 6x + x² - x³ - x⁴ + 2x⁵ - x⁶
# Note: u(x) = x² - x³, u'(x) = 2x - 3x², u''(x) = 2 - 6x, so -u'' = -2 + 6x
#       u(1-u) = (x²-x³)(1-x²+x³) = x² - x³ - x⁴ + 2x⁵ - x⁶

def source_term(x):
    """1D source term: f(x) = -2 + 6x + x² - x³ - x⁴ + 2x⁵ - x⁶"""
    # x shape: (..., 1) or (...,)
    x_flat = x.flatten() if x.ndim > 1 else x
    return (-2.0 + 6.0 * x_flat + x_flat**2 - x_flat**3 
            - x_flat**4 + 2.0 * x_flat**5 - x_flat**6)

def exact_solution(x):
    """1D exact solution: u(x) = x²(1-x)"""
    # x shape: (..., 1) or (...,)
    x_flat = x.flatten() if x.ndim > 1 else x
    return x_flat**2 * (1.0 - x_flat)

def boundary_value(x):
    """Dirichlet boundary condition: u(boundary) = exact_solution(x)"""
    return exact_solution(x)

# -------------------- PINN Loss (with stacked residuals) --------------------
def pde_residual(params, x_col):
    """
    Compute PDE residual: -d²u/dx² + u(1-u) - f(x) for 1D Fisher equation
    Returns residual vector of shape (n_col,)
    """
    def u_fn(x):
        """u(x) where x is a scalar (0D array)"""
        return forward(params, jnp.array([[x]])).squeeze()
    
    def fisher_operator(x):
        """Compute Fisher operator: -d²u/dx² + u(1-u)"""
        u = u_fn(x)
        u_xx = jax.grad(jax.grad(u_fn))(x)
        return -u_xx + u * (1.0 - u)
    
    # Compute Fisher operator for each collocation point
    # x_col shape: (n_col, 1), flatten to (n_col,) for vmap
    x_col_flat = x_col.reshape(-1)  # shape: (n_col,)
    pde_val = jax.vmap(fisher_operator)(x_col_flat)  # shape: (n_col,)
    f_val = source_term(x_col)  # shape: (n_col,)
    residual = pde_val - f_val
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

def eigs_top_k(JTJ, k):
    """Compute top k largest eigenvalues and eigenvectors."""
    S = 0.5 * (JTJ + JTJ.T)
    w, V = jnp.linalg.eigh(S)
    k_actual = min(k, w.size)
    if k_actual == 0:
        return jnp.array([], dtype=w.dtype), V[:, :0]
    lam = w[-k_actual:]
    Q = V[:, -k_actual:]
    return lam, Q

def apply_JTJ_eigs_to_vec(lam, Q, v):
    """Apply low-rank Hessian: H @ v where H = Q @ diag(λ) @ Q^T"""
    if lam.size == 0:
        return jnp.zeros_like(v)
    alpha = Q.T @ v
    return Q @ (lam * alpha)

# -------------------- Data (collocation + boundary) -------------------
# 1D domain partition: EDGES = [x_min, edge1, edge2, ..., x_max]
# Client i owns interval [EDGES[i], EDGES[i+1]]
edges_np = np.asarray(EDGES, dtype=np.float64)
CID = rank
L, R = float(edges_np[CID]), float(edges_np[CID + 1])
x_min, x_max = float(EDGES[0]), float(EDGES[-1])

# Collocation points (interior) - 1D
# Client CID samples points in [L, R]
key = jax.random.PRNGKey(12345 + CID)
key, subkey = jax.random.split(key)
x_col = jax.random.uniform(subkey, (N_SAMPLES, 1), minval=L, maxval=R)

# Boundary points - 1D
# 1D has only 2 boundaries: left (x = x_min) and right (x = x_max)
# Client 0 handles left boundary, last client handles right boundary
x_bc_list = []
if L == x_min:  # Client 0: left boundary
    key, subkey = jax.random.split(key)
    x_bc_left = jnp.full((N_BOUNDARY, 1), x_min)
    x_bc_list.append(x_bc_left)

if R == x_max:  # Last client: right boundary
    key, subkey = jax.random.split(key)
    x_bc_right = jnp.full((N_BOUNDARY, 1), x_max)
    x_bc_list.append(x_bc_right)

# Concatenate boundary points
if len(x_bc_list) > 0:
    x_bc = jnp.concatenate(x_bc_list, axis=0)
else:
    x_bc = jnp.zeros((0, 1), dtype=jnp.float64)

x_col_dev = device_put(x_col, DEV)
x_bc_dev = device_put(x_bc, DEV)

n_bc_local = x_bc.shape[0]
n_bc_total = comm.allreduce(n_bc_local, op=MPI.SUM)
n_bc_expected = ((1 if L == x_min else 0) + (1 if R == x_max else 0)) * N_BOUNDARY

if rank == 0:
    print(f"[data] 1D problem, N_CLIENTS={N_CLIENTS}")
    print(f"[data] domain partition: {EDGES}")
    print(f"[data] per-client collocation: {N_SAMPLES} points in [{L}, {R}]")
    print(f"[data] boundary points per client: {n_bc_expected} (expected), {n_bc_local} (actual)")
    print(f"[data] boundary points total: {n_bc_total} across all clients")
    print(f"[data] domain: [{x_min}, {x_max}]")

# Test points for evaluation - 1D uniform grid
x_min, x_max = float(EDGES[0]), float(EDGES[-1])
xs = jnp.linspace(x_min, x_max, GRID).reshape(-1, 1)
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

theta_ref, unravel_ref = ravel_pytree(params)
P = int(theta_ref.size)

# -------------------- Compile: stacked Jacobian (J̃ = [J_Ω; √λ_bc J_∂Ω]) --------------------
EMPTY_LOCAL = (x_col_dev.shape[0] == 0 and x_bc_dev.shape[0] == 0)

if not EMPTY_LOCAL:
    def stacked_residual(theta_vec):
        """
        Compute stacked residual: r̃ = [r_Ω; √λ_bc r_∂Ω]
        This is the key modification for PINN!
        """
        p = unravel_ref(theta_vec)
        r_pde = pde_residual(p, x_col_dev).reshape(-1)
        r_bc = boundary_residual(p, x_bc_dev).reshape(-1)
        # Stack with boundary weight
        r_stacked = jnp.concatenate([
            r_pde,
            jnp.sqrt(LAMBDA_BC) * r_bc
        ])
        return r_stacked
    
    B_total = int(x_col_dev.shape[0] + x_bc_dev.shape[0])
    base_jac = jacfwd(stacked_residual) if B_total >= P else jax.jacrev(stacked_residual)
    
    @jax.jit
    def jtj_and_g(theta_vec):
        """
        Compute J̃ᵀJ̃ and J̃ᵀr̃ where J̃ is the stacked Jacobian.
        This naturally gives H_i = J_Ω^T J_Ω + λ_bc J_∂Ω^T J_∂Ω
        """
        J_tilde = base_jac(theta_vec)  # Shape: (n_pde + n_bc, P)
        r_tilde = stacked_residual(theta_vec)  # Shape: (n_pde + n_bc,)
        
        # Compute second-order info
        H = J_tilde.T @ J_tilde  # This is the stacked H_i
        g = J_tilde.T @ r_tilde
        
        # Compute separate norms for monitoring
        n_pde = x_col_dev.shape[0]
        r_pde_norm = jnp.linalg.norm(r_tilde[:n_pde])
        r_bc_norm = jnp.linalg.norm(r_tilde[n_pde:]) / jnp.sqrt(LAMBDA_BC)
        r_total_norm = jnp.linalg.norm(r_tilde)
        
        return H, g, r_pde_norm, r_bc_norm, r_total_norm
    
    # Warm-up compilation
    _H, _g, _rpde, _rbc, _rtot = jtj_and_g(theta_ref)
    _H.block_until_ready(); _g.block_until_ready()
else:
    def jtj_and_g(theta_vec):
        return (jnp.zeros((P, P), theta_vec.dtype),
                jnp.zeros((P,), theta_vec.dtype),
                jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0))

if rank == 0:
    print("[jit] stacked (J̃ᵀJ̃, J̃ᵀr̃) kernel compiled for PINN.")

# ===== Warm-up =====
comm.Barrier()
opt_state_warm = optimizer.init(params)
p_warm, _ = train_step(params, opt_state_warm, x_col_dev, x_bc_dev)
jax.tree_util.tree_map(lambda a: a.block_until_ready(), p_warm)
_ = forward(params, xs)
_.block_until_ready()
if not EMPTY_LOCAL:
    H0_warm, _, _, _, _ = jtj_and_g(theta_ref)
    H0_warm.block_until_ready()
    _, _ = eigs_top_k(H0_warm, min(TOP_K, P))
if rank == 0:
    H0 = jnp.zeros((P, P), dtype=theta_ref.dtype)
    b0 = jnp.zeros((P,), dtype=theta_ref.dtype)
    _ = lstsq(H0, b0, rcond=PINV_TOL)[0]
    _.block_until_ready()
comm.Barrier()

    # 初始化日志
if rank == 0:
    with open(os.path.join(out_dir, "global_test.csv"), "w") as f:
        f.write("round,rel_l2_error\n")
    with open(os.path.join(out_dir, "timings.csv"), "w") as f:
        f.write("round,tJJ_max,tEig_max,tLocal_max,tComm_max,tRecon,tPinv,tUpdate,tBcast_max,tRound,comm_total,local_total,mwfl_total\n")

# -------------------- Training Loop --------------------
time_used_total = 0.0
for r in range(1, MAX_ROUNDS + 1):
    comm.Barrier()
    t_round0 = time.perf_counter()
    
    theta0_vec, unravel0 = ravel_pytree(params)
    
    # ---- Compute stacked JᵀJ and gradient ----
    t0 = time.perf_counter()
    JTJ, g_i, r_pde_norm, r_bc_norm, r_total_norm = jtj_and_g(theta0_vec)
    JTJ.block_until_ready(); g_i.block_until_ready()
    t_jj = time.perf_counter() - t0
    
    g_norm = float(jnp.linalg.norm(g_i))
    
    # ---- Local training (Adam) ----
    t2 = time.perf_counter()
    p_i = params
    opt_state = optimizer.init(params)
    for _ in range(LOCAL_EPOCHS):
        p_i, opt_state = train_step(p_i, opt_state, x_col_dev, x_bc_dev)
    th_i, _ = ravel_pytree(p_i)
    th_i.block_until_ready()
    t_local = time.perf_counter() - t2
    delta_i = th_i - theta0_vec
    
    tJJ_max = comm.allreduce(t_jj, op=MPI.MAX)
    tLocal_max = comm.allreduce(t_local, op=MPI.MAX)
    
    # ---- Eigen decomposition (low-rank approximation) ----
    t1 = time.perf_counter()
    lam, Q = eigs_top_k(JTJ, TOP_K)
    lam.block_until_ready(); Q.block_until_ready()
    t_eig = time.perf_counter() - t1
    tEig_max = comm.allreduce(t_eig, op=MPI.MAX)
    
    # ---- Gather low-rank representation ----
    t_c0 = time.perf_counter()
    lam_host = np.asarray(device_get(lam))
    Q_host = np.asarray(device_get(Q))
    delta_host = np.asarray(device_get(delta_i))
    payload = {"lam": lam_host, "Q": Q_host, "d": delta_host}
    gathered = comm.gather(payload, root=0)
    t_comm = time.perf_counter() - t_c0
    tComm_max = comm.allreduce(t_comm, op=MPI.MAX)
    
    if rank == 0:
        # ---- Reconstruct and aggregate H = Σ H_i ----
        t3 = time.perf_counter()
        P32 = jnp.float32
        H_sum = jnp.zeros((P, P), dtype=P32)
        b_sum = jnp.zeros((P,), dtype=P32)
        H_sum.block_until_ready(); b_sum.block_until_ready()
        for item in gathered:
            lam_i = device_put(jnp.asarray(item["lam"], dtype=P32), DEV)
            Q_i = device_put(jnp.asarray(item["Q"], dtype=P32), DEV)
            d_i = device_put(jnp.asarray(item["d"], dtype=P32), DEV)
            if lam_i.size > 0:
                H_i = Q_i @ (lam_i[:, None] * Q_i.T)
                H_sum = H_sum + H_i
                b_sum = b_sum + apply_JTJ_eigs_to_vec(lam_i, Q_i, d_i)
        H_sum.block_until_ready(); b_sum.block_until_ready()
        t_recon = time.perf_counter() - t3
        
        # ---- Solve linear system ----
        t4 = time.perf_counter()
        delta_flat = lstsq(H_sum, b_sum, rcond=PINV_TOL)[0]
        delta_flat.block_until_ready()
        t_pinv = time.perf_counter() - t4
        
        # ---- Global update ----
        t5 = time.perf_counter()
        theta_next = theta0_vec + STEP_SIZE * delta_flat
        theta_next.block_until_ready()
        params = unravel0(theta_next)
        t_update = time.perf_counter() - t5
    else:
        t_recon = t_pinv = t_update = 0.0
    
    # ---- Broadcast new params ----
    t_b0 = time.perf_counter()
    params_host = comm.bcast(tree_to_host(params) if rank == 0 else None, root=0)
    params = tree_to_device(params_host)
    t_bcast = time.perf_counter() - t_b0
    tBcast_max = comm.allreduce(t_bcast, op=MPI.MAX)
    
    if rank == 0:
        t_round = time.perf_counter() - t_round0
        time_this_round = float(
            tJJ_max + tEig_max + tLocal_max + tComm_max + t_recon + t_pinv + t_update + tBcast_max
        )
        time_used_total += time_this_round
        
        comm_total = tComm_max + tBcast_max
        local_total = tLocal_max
        mwfl_total = tJJ_max + tEig_max + t_recon + t_pinv + t_update
        
        # ---- Evaluation (Relative L2 error on test grid) ----
        u_pred = forward(params, xs).squeeze()
        u_true = ys_true.squeeze()
        # Relative L2 error: ||u_pred - u_true||_2 / ||u_true||_2
        l2_norm_error = jnp.sqrt(jnp.mean((u_pred - u_true) ** 2))
        l2_norm_true = jnp.sqrt(jnp.mean(u_true ** 2))
        rel_l2_error = float(l2_norm_error / l2_norm_true)
        
        print(f"[round {r:03d}] rel_L2_error={rel_l2_error:.6e} | "
              f"JJ(max)={tJJ_max:.3f}s, eig(max)={tEig_max:.3f}s, "
              f"comm={comm_total:.3f}s, local={local_total:.3f}s, mwfl={mwfl_total:.3f}s | "
              f"used={time_used_total:.1f}/{TIME_BUDGET:.1f}s")
        
        with open(os.path.join(out_dir, "global_test.csv"), "a") as f:
            f.write(f"{r},{rel_l2_error:.10e}\n")
        with open(os.path.join(out_dir, "timings.csv"), "a") as f:
            f.write(
                f"{r},{tJJ_max:.6f},{tEig_max:.6f},{tLocal_max:.6f},{tComm_max:.6f},"
                f"{t_recon:.6f},{t_pinv:.6f},{t_update:.6f},{tBcast_max:.6f},{t_round:.6f},"
                f"{comm_total:.6f},{local_total:.6f},{mwfl_total:.6f}\n"
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
    x_host = np.asarray(device_get(xs)).reshape(-1)  # shape: (n_points,)
    y_true_host = np.asarray(device_get(ys_true)).reshape(-1)
    y_pred_host = np.asarray(device_get(y_pred)).reshape(-1)
    
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w") as f:
        # Header: x, u_exact, u_pred
        f.write("x,u_exact,u_pred\n")
        for i in range(x_host.shape[0]):
            f.write(f"{x_host[i]:.10e},{y_true_host[i]:.10e},{y_pred_host[i]:.10e}\n")
    
    # Save final model parameters
    def to_numpy(x):
        if isinstance(x, jnp.ndarray):
            return np.asarray(device_get(x))
        return x
    
    params_numpy = jax.tree_util.tree_map(to_numpy, params)
    params_file = os.path.join(out_dir, "final_params.pkl")
    with open(params_file, 'wb') as f:
        pickle.dump(params_numpy, f)
    
    print("\n✓ PINN training completed!")
    print("Saved logs to:")
    print(" -", os.path.join(out_dir, "global_test.csv"))
    print(" -", os.path.join(out_dir, "timings.csv"))
    print(" -", pred_path)
    print(" -", params_file)

