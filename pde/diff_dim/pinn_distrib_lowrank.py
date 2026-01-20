# -*- coding: utf-8 -*-
"""Distributed low-rank PINN for PDE with boundary conditions."""

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
    print(f"[config] PINN low-rank: d_dim={D_DIM}, lr={LR}, local_epochs={LOCAL_EPOCHS}, step_size={STEP_SIZE}")
    print(f"[config] top_k={TOP_K}, lambda_bc={LAMBDA_BC}, n_boundary={N_BOUNDARY}, layers={LAYERS}")

# -------------------- 输出目录 --------------------
run_name = (
    f"pinn_lowrank_d{D_DIM}D_tb-{int(TIME_BUDGET)}s_mr-{MAX_ROUNDS}"
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

# -------------------- PINN Loss (with stacked residuals) --------------------
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

@jax.jit
def eigs_top_k(JTJ):
    """Compute top k largest eigenvalues and eigenvectors. JIT compiled for performance.
    Uses global TOP_K constant.
    
    Args:
        JTJ: P×P matrix
    """
    S = 0.5 * (JTJ + JTJ.T)
    w, V = jnp.linalg.eigh(S)
    # Use global TOP_K constant (fixed value)
    # w.shape[0] is Python int, TOP_K is Python int, so min() works
    n = w.shape[0]
    k_actual = min(TOP_K, n)
    # k_actual is Python int (compile-time constant), so Python if works
    if k_actual == 0:
        return jnp.array([], dtype=w.dtype), V[:, :0]
    lam = w[-k_actual:]
    Q = V[:, -k_actual:]
    return lam, Q

@jax.jit
def apply_JTJ_eigs_to_vec(lam, Q, v):
    """Apply low-rank Hessian: H @ v where H = Q @ diag(λ) @ Q^T. JIT compiled for performance."""
    if lam.size == 0:
        return jnp.zeros_like(v)
    alpha = Q.T @ v
    return Q @ (lam * alpha)

# -------------------- Data (collocation + boundary) -------------------
# Domain partition strategy:
# - EDGES defines partition of the FIRST dimension: [x_min, edge1, edge2, ..., x_max]
# - For d-dimensional domain: each client owns a "strip" region
#   * Client i: x_1 in [EDGES[i], EDGES[i+1]], x_2,...,x_d in [x_min, x_max]
# - Example: EDGES=[0,0.25,0.5,0.75,1.0] with d=2
#   * Client 0: x_1 in [0,0.25], x_2 in [0,1] (entire 2D strip)
#   * Client 1: x_1 in [0.25,0.5], x_2 in [0,1]
#   * Client 2: x_1 in [0.5,0.75], x_2 in [0,1]
#   * Client 3: x_1 in [0.75,1.0], x_2 in [0,1]
edges_np = np.asarray(EDGES, dtype=np.float64)
CID = rank
L, R = float(edges_np[CID]), float(edges_np[CID + 1])
x_min, x_max = float(EDGES[0]), float(EDGES[-1])

# Collocation points (interior) - d-dimensional
# For client CID: sample points in its assigned strip region
# - x_1 (first dimension) in [L, R] (client's partition)
# - x_2, ..., x_d (other dimensions) in [x_min, x_max] (full domain)
key = jax.random.PRNGKey(12345 + CID)
if D_DIM == 1:
    # 1D case: only first dimension
    key, subkey = jax.random.split(key)
    x_col = jax.random.uniform(subkey, (N_SAMPLES, 1), minval=L, maxval=R)
else:
    # d-dim case: first dimension in [L,R], others in [x_min, x_max]
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    # Generate first dimension in client's partition
    x_col_dim1 = jax.random.uniform(subkey1, (N_SAMPLES, 1), minval=L, maxval=R)
    # Generate remaining dimensions in full domain
    x_col_rest = jax.random.uniform(subkey2, (N_SAMPLES, D_DIM - 1), minval=x_min, maxval=x_max)
    # Concatenate to form d-dimensional points
    x_col = jnp.concatenate([x_col_dim1, x_col_rest], axis=1)

# Boundary points: d-dimensional boundaries
# Strategy: Each client handles boundary faces that intersect with its region (union of intersections)
# - Global domain: [x_min, x_max]^d has 2*d boundary faces
#   * For each dimension i: left face (x_i = x_min) and right face (x_i = x_max)
# - Client region: x_1 in [L, R], x_2,...,x_d in [x_min, x_max]
# - Client should handle boundary faces that intersect its region
#
# For Client CID:
# 1. First dimension (x_1):
#    - Left boundary (x_1 = x_min): if L == x_min (i.e., CID == 0)
#    - Right boundary (x_1 = x_max): if R == x_max (i.e., CID == N_CLIENTS-1)
# 2. Other dimensions (x_2,...,x_d):
#    - Both left (x_i = x_min) and right (x_i = x_max) boundaries intersect client region
#    - Sample on these boundaries with x_1 constrained to [L, R]
key_bc = key  # Save key for boundary sampling
x_bc_list = []

# Boundary 1: Left boundary of first dimension (x_1 = x_min) - only for Client 0
if L == x_min:  # Equivalent to CID == 0
    key_bc, subkey = jax.random.split(key_bc)
    # x_1 = x_min, x_2,...,x_d random in [x_min, x_max]
    x_bc_dim1_left = jax.random.uniform(subkey, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_dim1_left = x_bc_dim1_left.at[:, 0].set(x_min)
    x_bc_list.append(x_bc_dim1_left)

# Boundary 2: Right boundary of first dimension (x_1 = x_max) - only for last client
if R == x_max:  # Equivalent to CID == N_CLIENTS - 1
    key_bc, subkey = jax.random.split(key_bc)
    # x_1 = x_max, x_2,...,x_d random in [x_min, x_max]
    x_bc_dim1_right = jax.random.uniform(subkey, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    x_bc_dim1_right = x_bc_dim1_right.at[:, 0].set(x_max)
    x_bc_list.append(x_bc_dim1_right)

# Boundaries 3-2*d: Other dimensions (x_2, x_3, ..., x_d)
# Each dimension has 2 faces (left and right), all intersect with client region
# For these boundaries: x_1 in [L, R] (client's region), x_i fixed (boundary value), others random
for dim_idx in range(1, D_DIM):  # Start from dimension 1 (0-indexed: x_2 is index 1)
    # Left boundary: x_{dim_idx+1} = x_min
    key_bc, subkey1 = jax.random.split(key_bc)
    key_bc, subkey2 = jax.random.split(key_bc)
    # Generate all coordinates
    x_bc_other_left = jax.random.uniform(subkey1, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    # Overwrite x_1 to be in client region [L, R]
    x_bc_other_left = x_bc_other_left.at[:, 0].set(
        jax.random.uniform(subkey2, (N_BOUNDARY,), minval=L, maxval=R)
    )
    # Fix boundary dimension to x_min
    x_bc_other_left = x_bc_other_left.at[:, dim_idx].set(x_min)
    x_bc_list.append(x_bc_other_left)
    
    # Right boundary: x_{dim_idx+1} = x_max
    key_bc, subkey1 = jax.random.split(key_bc)
    key_bc, subkey2 = jax.random.split(key_bc)
    x_bc_other_right = jax.random.uniform(subkey1, (N_BOUNDARY, D_DIM), minval=x_min, maxval=x_max)
    # Overwrite x_1 to be in client region [L, R]
    x_bc_other_right = x_bc_other_right.at[:, 0].set(
        jax.random.uniform(subkey2, (N_BOUNDARY,), minval=L, maxval=R)
    )
    # Fix boundary dimension to x_max
    x_bc_other_right = x_bc_other_right.at[:, dim_idx].set(x_max)
    x_bc_list.append(x_bc_other_right)

# Concatenate all boundary points
if len(x_bc_list) > 0:
    x_bc = jnp.concatenate(x_bc_list, axis=0)
else:
    # Should not happen for d >= 1, but handle edge case
    x_bc = jnp.zeros((0, D_DIM), dtype=jnp.float64)

x_col_dev = device_put(x_col, DEV)
x_bc_dev = device_put(x_bc, DEV)

n_bc_local = x_bc.shape[0]
n_bc_total = comm.allreduce(n_bc_local, op=MPI.SUM)
# Calculate expected boundary points per client
# Each client handles: 
#   - First dim boundaries: 0 or 1 or 2 (depending on position)
#   - Other dim boundaries: 2*(d-1) (left and right for each of d-1 dimensions)
#   - Total per boundary: N_BOUNDARY points
n_bc_first_dim = (1 if L == x_min else 0) + (1 if R == x_max else 0)
n_bc_other_dims = 2 * (D_DIM - 1)  # 2 faces per dimension for dimensions 2..d
n_bc_expected = (n_bc_first_dim + n_bc_other_dims) * N_BOUNDARY

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
# Use random points instead of grid for better coverage in high dimensions
x_min, x_max = float(EDGES[0]), float(EDGES[-1])
key_test = jax.random.PRNGKey(99999)  # Fixed key for reproducible test points
key_test, subkey_test = jax.random.split(key_test)
# Generate GRID random points uniformly in d-dimensional hypercube [x_min, x_max]^d
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
    lam_warm, Q_warm = eigs_top_k(H0_warm)
    lam_warm.block_until_ready(); Q_warm.block_until_ready()
    # Warm-up apply_JTJ_eigs_to_vec function
    v_dummy = jnp.ones((P,), dtype=H0_warm.dtype)
    _ = apply_JTJ_eigs_to_vec(lam_warm, Q_warm, v_dummy)
    _.block_until_ready()

if rank == 0:
    # JIT compile server-side functions for performance
    @jax.jit
    def solve_linear_system(H, b):
        """JIT compiled linear system solver."""
        return lstsq(H, b, rcond=PINV_TOL)[0]
    
    @jax.jit
    def global_update(theta0, delta, step_size):
        """JIT compiled global parameter update."""
        return theta0 + step_size * delta
    
    # JIT compiled function for reconstructing H_i and computing H_i @ d_i
    @jax.jit
    def reconstruct_H_and_b(lam_i, Q_i, d_i):
        """
        Reconstruct H_i = Q_i @ diag(λ_i) @ Q_i^T and compute H_i @ d_i.
        Returns (H_i, H_i @ d_i).
        """
        if lam_i.size > 0:
            H_i = Q_i @ (lam_i[:, None] * Q_i.T)
            b_i = apply_JTJ_eigs_to_vec(lam_i, Q_i, d_i)
        else:
            H_i = jnp.zeros((P, P), dtype=Q_i.dtype)
            b_i = jnp.zeros((P,), dtype=d_i.dtype)
        return H_i, b_i
    
    # Warm-up compilation
    H0 = jnp.zeros((P, P), dtype=theta_ref.dtype)
    b0 = jnp.zeros((P,), dtype=theta_ref.dtype)
    _ = solve_linear_system(H0, b0)
    _.block_until_ready()
    _ = global_update(theta_ref, b0, STEP_SIZE)
    _.block_until_ready()
    # Warm-up reconstruct function (use dummy data)
    if not EMPTY_LOCAL:
        lam_dummy = jnp.ones((min(TOP_K, P),), dtype=jnp.float32)
        Q_dummy = jnp.ones((P, min(TOP_K, P)), dtype=jnp.float32)
        d_dummy = jnp.ones((P,), dtype=jnp.float32)
        _, _ = reconstruct_H_and_b(lam_dummy, Q_dummy, d_dummy)
        _.block_until_ready()
else:
    solve_linear_system = None
    global_update = None
    reconstruct_H_and_b = None
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
    lam, Q = eigs_top_k(JTJ)
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
        # ---- Reconstruct and aggregate H = Σ H_i (JIT optimized) ----
        t3 = time.perf_counter()
        P32 = jnp.float32
        H_sum = jnp.zeros((P, P), dtype=P32)
        b_sum = jnp.zeros((P,), dtype=P32)
        H_sum.block_until_ready(); b_sum.block_until_ready()
        for item in gathered:
            # Convert to device arrays (outside JIT)
            lam_i = device_put(jnp.asarray(item["lam"], dtype=P32), DEV)
            Q_i = device_put(jnp.asarray(item["Q"], dtype=P32), DEV)
            d_i = device_put(jnp.asarray(item["d"], dtype=P32), DEV)
            # Use JIT compiled function for core computation
            H_i, b_i = reconstruct_H_and_b(lam_i, Q_i, d_i)
            H_sum = H_sum + H_i
            b_sum = b_sum + b_i
        H_sum.block_until_ready(); b_sum.block_until_ready()
        t_recon = time.perf_counter() - t3
        
        # ---- Solve linear system (JIT compiled) ----
        t4 = time.perf_counter()
        delta_flat = solve_linear_system(H_sum, b_sum)
        delta_flat.block_until_ready()
        t_pinv = time.perf_counter() - t4
        
        # ---- Global update (JIT compiled) ----
        t5 = time.perf_counter()
        theta_next = global_update(theta0_vec, delta_flat, STEP_SIZE)
        theta_next.block_until_ready()
        params = unravel0(theta_next)
        # Ensure params are ready before broadcast (fixes broadcast timing issue)
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), params)
        # Convert to host before timing (device_get is slow, should not be in broadcast time)
        params_host_for_bcast = tree_to_host(params) if rank == 0 else None
        t_update = time.perf_counter() - t5
    else:
        t_recon = t_pinv = t_update = 0.0
        params_host_for_bcast = None
    
    # ---- Broadcast new params ----
    # Only measure MPI communication time, not data conversion
    t_b0 = time.perf_counter()
    params_host = comm.bcast(params_host_for_bcast, root=0)
    t_bcast = time.perf_counter() - t_b0
    tBcast_max = comm.allreduce(t_bcast, op=MPI.MAX)
    # Convert to device after timing (device_put is slow, should not be in broadcast time)
    params = tree_to_device(params_host)
    
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
    x_host = np.asarray(device_get(xs))  # shape: (n_points, d_dim)
    y_true_host = np.asarray(device_get(ys_true)).reshape(-1)
    y_pred_host = np.asarray(device_get(y_pred)).reshape(-1)
    
    pred_path = os.path.join(out_dir, "predictions.csv")
    with open(pred_path, "w") as f:
        # Header: x1, x2, ..., xd, u_exact, u_pred
        header = ",".join([f"x{i+1}" for i in range(D_DIM)]) + ",u_exact,u_pred\n"
        f.write(header)
        for i in range(x_host.shape[0]):
            x_row = x_host[i]  # shape: (d_dim,)
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
    
    print("\n✓ PINN training completed!")
    print("Saved logs to:")
    print(" -", os.path.join(out_dir, "global_test.csv"))
    print(" -", os.path.join(out_dir, "timings.csv"))
    print(" -", pred_path)
    print(" -", params_file)

