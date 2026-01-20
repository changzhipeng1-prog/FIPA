"""QR-based aggregation with checkpoint loading."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PARENT_ROOT))

import os
import time
import argparse
import csv
import random
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from flax.training import train_state
import optax

from models_jax import (
    create_model,
    init_model,
    create_medium_model_20k,
    create_large_model_200k,
    create_resnet20,
)
from train_jax import train_local_model, evaluate_model, params_to_vector, vector_to_params
from data_jax import load_cifar10_data
from jax.flatten_util import ravel_pytree


os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)  # Aligned with FedRCL (seed=0)
jax.config.update('jax_default_prng_impl', 'rbg')
jax.config.update('jax_enable_x64', False)  # Single precision (float32), aligned with PyTorch
np.random.seed(0)  # Aligned with FedRCL (seed=0)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def streaming_matvec_H(params, model, X_client, y_client, V, block_size=128):
    """Compute H*V for the empirical Fisher/Hessian using streaming mini-batches."""
    @jax.jit
    def compute_jacobian_matvec_block(params, x_block, mask, v_mat):
        logits_block = model.apply(params, x_block, training=False)

        def compute_single(x, logits, m):
            def grad_j(j):
                grad_tree = jax.grad(lambda p: model.apply(p, x[None, ...], training=False)[0][j])(params)
                grad_flat, _ = ravel_pytree(grad_tree)
                return grad_flat

            J_n = jax.vmap(grad_j)(jnp.arange(logits.shape[0]))
            probs = jax.nn.softmax(logits)

            J_v = J_n @ v_mat
            probs_J_v = probs[:, None] * J_v
            probs_outer = jnp.dot(probs, J_v) * probs[:, None]
            S_J_v = probs_J_v - probs_outer

            return m * (J_n.T @ S_J_v)

        batch_results = jax.vmap(compute_single)(x_block, logits_block, mask)
        return batch_results.sum(axis=0)

    N = X_client.shape[0]
    p, ell = V.shape
    H_V = jnp.zeros((p, ell))

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        current = X_client[start:end]
        block_len = end - start
        if block_len < block_size:
            pad_len = block_size - block_len
            pad_block = np.repeat(current[-1:], pad_len, axis=0)
            block_np = np.concatenate([current, pad_block], axis=0)
            mask_np = np.concatenate(
                [np.ones(block_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)], axis=0
            )
        else:
            block_np = current
            mask_np = np.ones(block_size, dtype=np.float32)

        x_block = jnp.asarray(block_np)
        mask_block = jnp.asarray(mask_np)
        H_V += compute_jacobian_matvec_block(params, x_block, mask_block, V)

    H_V = H_V / N
    return H_V


def subspace_iteration_rayleigh_ritz(params, model, X_client, y_client, k, s=10, q=1, round_idx=None, block_size=128):
    """Estimate top-k eigenpairs of the Fisher/Hessian via randomized subspace iteration."""
    param_flat, _ = ravel_pytree(params)
    p = len(param_flat)
    ell = k + s

    seed = 42 + (round_idx * 1000 if round_idx is not None else 0)
    rng = jax.random.PRNGKey(seed)

    V = jax.random.normal(rng, (p, ell))
    V, _ = jnp.linalg.qr(V, mode='reduced')

    for _ in range(q):
        W = streaming_matvec_H(params, model, X_client, y_client, V, block_size=block_size)
        V, _ = jnp.linalg.qr(W, mode='reduced')

    H_V = streaming_matvec_H(params, model, X_client, y_client, V, block_size=block_size)
    S_small = V.T @ H_V
    S_small = (S_small + S_small.T) / 2

    eigenvals_small, eigenvecs_small = jnp.linalg.eigh(S_small)
    idx = jnp.argsort(eigenvals_small)[::-1]
    eigenvals_small = eigenvals_small[idx]
    eigenvecs_small = eigenvecs_small[:, idx]

    positive_mask = eigenvals_small > 1e-8
    eigenvals_pos = eigenvals_small[positive_mask]
    eigenvecs_pos = eigenvecs_small[:, positive_mask]

    k_actual = min(k, len(eigenvals_pos))
    eigenvals_k = eigenvals_pos[:k_actual]
    eigenvecs_small_k = eigenvecs_pos[:, :k_actual]
    eigenvecs_k = V @ eigenvecs_small_k

    return eigenvals_k, eigenvecs_k


def sample_active_clients(total_clients, client_fraction, seed, num_gpus=4):
    """Randomly sample a subset of clients ensuring no GPU collision."""
    active_count = max(1, int(total_clients * client_fraction))
    active_count = min(active_count, num_gpus)  # Cannot exceed number of GPUs
    
    rng = np.random.default_rng(seed)
    
    # Group clients by their GPU assignment (rank % num_gpus)
    gpu_groups = [[] for _ in range(num_gpus)]
    for client_id in range(total_clients):
        gpu_id = client_id % num_gpus
        gpu_groups[gpu_id].append(client_id)
    
    # Sample one client from each GPU group to avoid collision
    active_indices = []
    for gpu_id in range(active_count):
        if gpu_groups[gpu_id]:
            selected = rng.choice(gpu_groups[gpu_id])
            active_indices.append(selected)
    
    return sorted(active_indices)


def parse_checkpoint_path(checkpoint_path):
    """Extract model_size and alpha from checkpoint path.
    
    Expected path format: .../model_{model_size}_a{alpha}_lr{lr}_le{le}_w{w}/checkpoint_round_{round}.pkl
    Returns: (model_size, alpha) or (None, None) if cannot parse
    """
    checkpoint_path_obj = Path(checkpoint_path)
    parent_dir = checkpoint_path_obj.parent.name
    
    # Try to parse: model_{model_size}_a{alpha}_lr...
    if parent_dir.startswith('model_'):
        parts = parent_dir.split('_')
        model_size = None
        alpha = None
        
        # Find model_size (after 'model_')
        for i, part in enumerate(parts):
            if part == 'model' and i + 1 < len(parts):
                model_size = parts[i + 1]
                break
        
        # Find alpha (starts with 'a' followed by number)
        for part in parts:
            if part.startswith('a') and len(part) > 1:
                try:
                    alpha = float(part[1:])
                    break
                except ValueError:
                    continue
        
        # Map model_size names
        if model_size:
            if model_size == 'resnet20':
                pass  # Keep as is
            elif model_size in ['light', 'medium20k', 'large200k']:
                pass  # Keep as is
            else:
                # Try alternative names
                if 'resnet20' in parent_dir or 'resnet' in parent_dir:
                    model_size = 'resnet20'
                elif 'large200k' in parent_dir or 'large' in parent_dir:
                    model_size = 'large200k'
                elif 'medium20k' in parent_dir or 'medium' in parent_dir:
                    model_size = 'medium20k'
                elif 'light' in parent_dir:
                    model_size = 'light'
                else:
                    model_size = None
        
        return model_size, alpha
    
    return None, None


def load_checkpoint(checkpoint_path):
    """Load model parameters and metadata from checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Handle both old format (just params) and new format (dict with metadata)
    if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
        metadata = checkpoint_data.copy()
    else:
        # Old format: just params, try to extract round from filename
        checkpoint_path_obj = Path(checkpoint_path)
        round_num = None
        if 'round_' in checkpoint_path_obj.stem:
            try:
                round_num = int(checkpoint_path_obj.stem.split('round_')[1].split('.')[0])
            except:
                pass
        metadata = {'round': round_num, 'params': checkpoint_data}
        checkpoint_data = metadata
    
    # Try to extract model_size and alpha from path if not in metadata
    if 'model_size' not in metadata or 'alpha' not in metadata:
        model_size_from_path, alpha_from_path = parse_checkpoint_path(checkpoint_path)
        if model_size_from_path is not None:
            metadata['model_size_from_path'] = model_size_from_path
        if alpha_from_path is not None:
            metadata['alpha_from_path'] = alpha_from_path
    
    return checkpoint_data['params'] if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data else checkpoint_data, metadata


def main():
    """Entry point for QR aggregation starting from checkpoint."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        parser = argparse.ArgumentParser(description='QR aggregation with checkpoint loading')
        parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint file')
        parser.add_argument('--checkpoint_round', type=int, default=None, help='Round number of checkpoint (auto-detected if not provided)')
        parser.add_argument('--initial_lr', type=float, default=None, help='Initial LR (auto-loaded from checkpoint if available)')
        parser.add_argument('--qr_rounds', type=int, default=50, help='QR-mix rounds')
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--total_clients', type=int, default=100)  # Aligned with train_fedavg_only.py
        parser.add_argument('--client_fraction', type=float, default=0.05)  # Aligned with train_fedavg_only.py
        parser.add_argument('--rank_k', type=int, default=200)
        parser.add_argument('--oversampling', type=int, default=50)
        parser.add_argument('--power_iter', type=int, default=5)
        parser.add_argument('--output_dir', type=str, default='qr_results')
        parser.add_argument('--block_size', type=int, default=64)
        parser.add_argument('--beta', type=float, default=0.5)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--local_epochs', type=int, default=5)
        parser.add_argument('--client_selection_seed', type=int, default=0)  # Aligned with train_fedavg_only.py
        parser.add_argument('--model_size', type=str, default='resnet20', choices=['light', 'medium20k', 'large200k', 'resnet20'])
        parser.add_argument('--eval_batch_size', type=int, default=128)  # Aligned with FedRCL
        parser.add_argument('--train_batch_size', type=int, default=50)  # Aligned with FedRCL
        parser.add_argument('--weight_decay', type=float, default=1e-3)  # Aligned with FedRCL
        parser.add_argument('--lr_decay', type=float, default=None, help='LR decay rate (auto-loaded from checkpoint if available)')
        parser.add_argument('--start_round', type=int, default=None, help='Starting round number (auto-detected from checkpoint if not provided)')
        parser.add_argument('--lr_scale', type=float, default=1.0, help='Learning rate scaling factor. Applied to the computed LR from checkpoint. Default: 1.0 (no scaling). Use 0.1 to divide LR by 10.')
        args = parser.parse_args()
        config = vars(args)
    else:
        config = None

    config = comm.bcast(config, root=0)

    checkpoint_path = config['checkpoint_path']
    checkpoint_round_arg = config.get('checkpoint_round')
    initial_lr_arg = config.get('initial_lr')
    lr_decay_arg = config.get('lr_decay')
    start_round_arg = config.get('start_round')
    lr_scale = config.get('lr_scale', 1.0)
    qr_rounds = config['qr_rounds']
    alpha_arg = config['alpha']
    total_clients = config['total_clients']
    client_fraction = config['client_fraction']
    rank_k = config['rank_k']
    s = config['oversampling']
    q = config['power_iter']
    results_dir = config['output_dir']
    block_size = config['block_size']
    beta_reg = float(config['beta'])
    gamma_scale = float(config['gamma'])
    local_epochs = int(config['local_epochs'])
    client_selection_seed = config['client_selection_seed']
    model_size_arg = config['model_size']
    eval_batch_size = config['eval_batch_size']
    train_batch_size = config.get('train_batch_size', 50)
    weight_decay = config.get('weight_decay', 1e-3)

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        device_list = [dev.strip() for dev in visible_devices.split(',') if dev.strip()]
        if device_list:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_list[rank % len(device_list)]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 8)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 8)

    num_gpus = size
    
    # Try to infer model_size and alpha from checkpoint path and validate with user args
    model_size_from_path, alpha_from_path = parse_checkpoint_path(checkpoint_path)
    
    # Use path-inferred values if available, otherwise use command line args
    if rank == 0:
        if model_size_from_path is not None:
            if model_size_arg != model_size_from_path:
                print(f'⚠️  Warning: --model_size={model_size_arg} does not match checkpoint path (inferred: {model_size_from_path})')
                print(f'   Using inferred model_size: {model_size_from_path}')
            model_size = model_size_from_path
        else:
            model_size = model_size_arg
            if model_size_arg == 'resnet20':  # Default value, might not be set explicitly
                print(f'⚠️  Warning: Could not infer model_size from checkpoint path. Using --model_size={model_size}')
        
        if alpha_from_path is not None:
            if abs(alpha_arg - alpha_from_path) > 1e-6:
                print(f'⚠️  Warning: --alpha={alpha_arg} does not match checkpoint path (inferred: {alpha_from_path})')
                print(f'   Using inferred alpha: {alpha_from_path}')
            alpha = alpha_from_path
        else:
            alpha = alpha_arg
            if alpha_arg == 0.1:  # Default value, might not be set explicitly
                print(f'⚠️  Warning: Could not infer alpha from checkpoint path. Using --alpha={alpha}')
    else:
        model_size = None
        alpha = None
    
    # Broadcast inferred values
    model_size = comm.bcast(model_size, root=0)
    alpha = comm.bcast(alpha, root=0)
    
    # Note: initial_lr, lr_decay, start_round will be set after loading checkpoint
    if rank == 0:
        print(f'Config: model={model_size}, alpha={alpha}, total_clients={total_clients}, fraction={client_fraction:.2f}, '
              f'qr_rounds={qr_rounds}, k={rank_k}, s={s}, q={q}, '
              f'beta={beta_reg}, gamma={gamma_scale}, local_epochs={local_epochs}, lr_scale={lr_scale}')
        print(f'Checkpoint: {checkpoint_path}')
        print(f'GPU assignment: {num_gpus} ranks/GPUs serving {total_clients} logical clients (max {num_gpus} active per round)')

    # Load federated datasets
    # NOTE: Using the same random seed (0) and same parameters ensures
    #       the same data distribution as warmup training
    train_datasets, test_data, net_cls_counts = load_cifar10_data(total_clients, alpha, verbose=(rank==0))
    client_samples = np.array([len(data[0]) for data in train_datasets])
    weights = client_samples / client_samples.sum()

    if rank == 0:
        print(f'\n{"="*80}')
        print(f'Data Distribution Information')
        print(f'{"="*80}')
        print(f'Total clients: {total_clients}')
        print(f'Alpha (heterogeneity): {alpha}')
        print(f'Total training samples: {client_samples.sum()}')
        print(f'Average samples per client: {client_samples.mean():.1f}')
        print(f'Min/Max samples per client: {client_samples.min()}/{client_samples.max()}')
        print(f'Total test samples: {len(test_data[0])}')
        print(f'\nData is loaded with the SAME distribution as warmup training:')
        print(f'  - Same random seed: 0')
        print(f'  - Same alpha: {alpha}')
        print(f'  - Same total_clients: {total_clients}')
        print(f'  - Same data split (Dirichlet with balanced=True)')
        print(f'{"="*80}\n')

    if model_size == 'light':
        model = create_model(10)
    elif model_size == 'medium20k':
        model = create_medium_model_20k(10)
    elif model_size == 'large200k':
        model = create_large_model_200k(10)
    elif model_size == 'resnet20':
        model = create_resnet20(n_classes=10, use_bn_layer=False)
    else:
        raise ValueError(f'Unsupported model_size: {model_size}. Supported: light, medium20k, large200k, resnet20')

    # Load checkpoint and extract metadata
    if rank == 0:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        print(f'\nLoading checkpoint from: {checkpoint_path}')
        qr_mix_params, checkpoint_meta = load_checkpoint(checkpoint_path)
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract training configuration from checkpoint metadata
        checkpoint_round = checkpoint_round_arg if checkpoint_round_arg is not None else checkpoint_meta.get('round')
        if checkpoint_round is None:
            raise ValueError('Cannot determine checkpoint round. Please provide --checkpoint_round or ensure checkpoint contains round info.')
        
        # Use checkpoint metadata if available, otherwise use command line args
        initial_lr = initial_lr_arg if initial_lr_arg is not None else checkpoint_meta.get('initial_lr', 0.1)
        lr_decay = lr_decay_arg if lr_decay_arg is not None else checkpoint_meta.get('lr_decay', 0.998)
        start_round = start_round_arg if start_round_arg is not None else checkpoint_round
        
        # Verify consistency: start_round should be checkpoint_round (QR-mix starts from next round)
        if start_round != checkpoint_round:
            print(f'Warning: start_round ({start_round}) != checkpoint_round ({checkpoint_round}). Using checkpoint_round.')
            start_round = checkpoint_round
        
        # Calculate LR that will be used in QR-mix first round
        # QR-mix starts from round (checkpoint_round + 1)
        first_qr_round = checkpoint_round + 1
        first_qr_lr_base = initial_lr * (lr_decay ** first_qr_round)
        first_qr_lr = first_qr_lr_base * lr_scale
        checkpoint_lr = initial_lr * (lr_decay ** checkpoint_round)
        
        print(f'\n{"="*80}')
        print(f'Checkpoint Information')
        print(f'{"="*80}')
        print(f'Checkpoint path: {checkpoint_path}')
        print(f'Checkpoint metadata:')
        print(f'  Round: {checkpoint_round}')
        print(f'  Initial LR: {initial_lr}')
        print(f'  LR Decay: {lr_decay}')
        print(f'  LR Scale: {lr_scale}')
        print(f'  LR at checkpoint (round {checkpoint_round}): {checkpoint_lr:.6f}')
        print(f'  QR-mix will start from round {first_qr_round} with LR {first_qr_lr:.6f} (base: {first_qr_lr_base:.6f} × scale: {lr_scale})')
        
        # Evaluate initial checkpoint
        print(f'\nEvaluating checkpoint model...')
        eval_start = time.time()
        checkpoint_acc, checkpoint_loss = evaluate_model(qr_mix_params, model.apply, test_data, batch_size=eval_batch_size)
        eval_time = time.time() - eval_start
        print(f'{"="*80}')
        print(f'Checkpoint Evaluation Results:')
        print(f'  Test Accuracy: {checkpoint_acc:.4f}%')
        print(f'  Test Loss:     {checkpoint_loss:.6f}')
        print(f'  Evaluation time: {eval_time:.2f} seconds')
        print(f'{"="*80}\n')
    else:
        qr_mix_params = None
        checkpoint_round = None
        initial_lr = None
        lr_decay = None
        start_round = None
        checkpoint_acc = None
        checkpoint_loss = None

    # Broadcast checkpoint data and config
    # Note: lr_scale is already obtained from config (line 284), no need to broadcast separately
    qr_mix_params = comm.bcast(qr_mix_params, root=0)
    checkpoint_round = comm.bcast(checkpoint_round, root=0)
    initial_lr = comm.bcast(initial_lr, root=0)
    lr_decay = comm.bcast(lr_decay, root=0)
    start_round = comm.bcast(start_round, root=0)
    checkpoint_acc = comm.bcast(checkpoint_acc, root=0)
    checkpoint_loss = comm.bcast(checkpoint_loss, root=0)
    comm.Barrier()

    rng = jax.random.PRNGKey(0 + rank)  # Aligned with FedRCL and train_fedavg_only.py (seed=0)
    
    # Initialize results with checkpoint evaluation
    if rank == 0:
        results = [{
            'phase': 'checkpoint',
            'round': checkpoint_round,
            'qr_acc': float(checkpoint_acc),
            'qr_loss': float(checkpoint_loss),
            'qr_eig_time_sec': 0.0,
            'qr_train_time_sec': 0.0,
            'qr_agg_time_sec': 0.0,
        }]
    else:
        results = []

    # QR-mix Rounds
    # =================================================================
    if rank == 0:
        print(f'\n=== QR-mix Rounds ({qr_rounds} rounds) ===')

    for qr_round_idx in range(qr_rounds):
        total_round = start_round + qr_round_idx + 1
        
        # Learning rate decay (aligned with FedRCL: local_lr_decay=0.998 per round)
        # Important: total_round is the current round number
        # If checkpoint is at round N, QR-mix starts from round N+1
        # Round N+1 should use LR = initial_lr * (lr_decay ** (N+1)) * lr_scale
        # This ensures QR-mix uses the same LR as FedAvg would at the same round, scaled by lr_scale
        # Example: checkpoint at round 1000 -> QR-mix round 1 uses round 1001 with LR = initial_lr * (lr_decay ** 1001) * lr_scale
        current_lr_base = initial_lr * (lr_decay ** total_round)
        current_lr = current_lr_base * lr_scale
        
        if rank == 0:
            active_indices = sample_active_clients(total_clients, client_fraction, client_selection_seed + total_round, num_gpus)
            print(f'\n[QR Round] {qr_round_idx+1}/{qr_rounds} (Total Round {total_round}, LR={current_lr:.6f}), Active clients: {active_indices}')
        else:
            active_indices = None
        
        active_indices = comm.bcast(active_indices, root=0)
        active_set = set(active_indices)

        assigned_cid = None
        for cid in active_indices:
            if cid % size == rank:
                assigned_cid = cid
                break
        
        # QR-mix Path
        eigenvals_qr = np.zeros((0,))
        eigenvecs_qr = np.zeros((0, 0))
        
        if assigned_cid is not None:
            X_client, y_client = train_datasets[assigned_cid]
            # Compute Hessian eigenpairs
            h_start = time.time()
            eigenvals_jax, eigenvecs_jax = subspace_iteration_rayleigh_ritz(
                qr_mix_params, model, X_client, y_client, rank_k, s, q,
                round_idx=total_round, block_size=block_size
            )
            h_time = time.time() - h_start
            eigenvals_qr = np.asarray(eigenvals_jax)
            eigenvecs_qr = np.asarray(eigenvecs_jax)
            
            # Local training
            qr_train_start = time.time()
            rng, client_rng = jax.random.split(rng)
            # Optimizer: SGD with lr=current_lr, momentum=0.0, weight_decay=1e-3 (aligned with FedRCL)
            tx_qr = optax.chain(
                optax.add_decayed_weights(weight_decay),
                optax.sgd(learning_rate=current_lr, momentum=0.0)
            )
            state_qr = train_state.TrainState.create(
                apply_fn=model.apply,
                params=qr_mix_params,
                tx=tx_qr,
            )
            params_before_qr = params_to_vector(state_qr.params)
            state_qr, _, _ = train_local_model(state_qr, (X_client, y_client), local_epochs, train_batch_size, client_rng)
            params_after_qr = params_to_vector(state_qr.params)
            delta_w_qr = np.asarray(params_after_qr - params_before_qr)
            qr_train_time = time.time() - qr_train_start
        else:
            h_time = 0.0
            delta_w_qr = None
            qr_train_time = 0.0
        
        # Gather QR data
        client_delta_qr = (assigned_cid, delta_w_qr)
        all_deltas_qr = comm.gather(client_delta_qr, root=0)
        eigen_payload_qr = (assigned_cid, eigenvals_qr, eigenvecs_qr) if assigned_cid is not None else (None, None, None)
        eigen_payloads_qr = comm.gather(eigen_payload_qr, root=0)
        h_time_list = comm.gather(h_time, root=0)
        qr_train_time_list = comm.gather(qr_train_time, root=0)
        
        # Server aggregation (QR-mix)
        if rank == 0:
            agg_start = time.time()
            # Simple average instead of weighted average (aligned with FedRCL)
            num_active = len(active_indices)
            normalized_weights = {i: 1.0 / num_active for i in active_indices}
            
            # QR-mix aggregation
            qr_vec = params_to_vector(qr_mix_params)
            eigen_pairs = []
            eigenpair_dict = {}
            for cid, vals, vecs in eigen_payloads_qr:
                if cid is not None and vals is not None and vecs is not None:
                    eigenpair_dict[cid] = (jnp.array(vals), jnp.array(vecs))
            delta_qr_dict = {}
            for cid, delta in all_deltas_qr:
                if cid is not None and delta is not None:
                    delta_qr_dict[cid] = delta

            for idx in active_indices:
                if idx in eigenpair_dict:
                    vals, vecs = eigenpair_dict[idx]
                    eigen_pairs.append((idx, vals, vecs))
                else:
                    raise RuntimeError(f'Missing eigen data for active client {idx}')
            
            # Concatenate eigenvectors from active clients
            V_stacked = jnp.concatenate([vecs for (_, _, vecs) in eigen_pairs], axis=1)
            Sigma_weighted = jnp.concatenate([vals * normalized_weights[idx] for idx, vals, _ in eigen_pairs])
            
            # QR decomposition
            Q, R = jnp.linalg.qr(V_stacked, mode='reduced')
            K = R @ jnp.diag(Sigma_weighted) @ R.T
            K = (K + K.T) / 2
            
            # Compute b
            b = jnp.zeros_like(qr_vec)
            for idx, vals_i, vecs_i in eigen_pairs:
                if idx not in delta_qr_dict:
                    raise RuntimeError(f'Missing QR delta for active client {idx}')
                delta_i = jnp.array(delta_qr_dict[idx])
                b += normalized_weights[idx] * vecs_i @ (vals_i * (vecs_i.T @ delta_i))
            
            # Solve and update
            if Q.size != 0:
                proj = Q.T @ b
                eye_r = jnp.eye(K.shape[0], dtype=K.dtype)
                z = jnp.linalg.solve(K + beta_reg * eye_r, proj)
                delta_subspace = Q @ z
                residual = b - Q @ proj
                qr_delta = gamma_scale * (beta_reg * delta_subspace + residual)
            else:
                qr_delta = gamma_scale * b
            
            qr_mix_params = vector_to_params(qr_vec + qr_delta, qr_mix_params)
            qr_acc, qr_loss = evaluate_model(qr_mix_params, model.apply, test_data, batch_size=eval_batch_size)
            
            qr_agg_time_sec = time.time() - agg_start
            qr_eig_time_sec = max(h_time_list) if h_time_list else 0.0
            qr_train_time_sec = max(qr_train_time_list) if qr_train_time_list else 0.0
            print(f'  QR-mix Round {qr_round_idx+1}/{qr_rounds} (Total Round {total_round}, LR={current_lr:.6f}): Acc={qr_acc:.2f}%, Loss={qr_loss:.4f}')
            print(f'  Timings (sec) -> Eigen: {qr_eig_time_sec:.2f}, Train: {qr_train_time_sec:.2f}, Aggregate: {qr_agg_time_sec:.2f}')
            
            results.append({
                'phase': 'qr_mix',
                'round': total_round,
                'qr_acc': float(qr_acc),
                'qr_loss': float(qr_loss),
                'qr_eig_time_sec': float(qr_eig_time_sec),
                'qr_train_time_sec': float(qr_train_time_sec),
                'qr_agg_time_sec': float(qr_agg_time_sec),
            })
        
        qr_mix_params = comm.bcast(qr_mix_params, root=0)

    # Save results
    if rank == 0:
        csv_file = Path(results_dir) / 'training_results.csv'
        with csv_file.open('w', newline='') as f:
            if results:
                fieldnames = [
                    'phase',
                    'round',
                    'qr_acc',
                    'qr_loss',
                    'qr_eig_time_sec',
                    'qr_train_time_sec',
                    'qr_agg_time_sec',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    normalized_row = {field: row.get(field, '') for field in fieldnames}
                    writer.writerow(normalized_row)
        
        print(f'\n{"="*80}')
        print(f'Results Summary')
        print(f'{"="*80}')
        print(f'Checkpoint (round {checkpoint_round}): Acc={checkpoint_acc:.4f}%, Loss={checkpoint_loss:.6f}')
        if len(results) > 1:
            final_result = results[-1]
            print(f'Final QR-mix (round {final_result["round"]}): Acc={final_result["qr_acc"]:.4f}%, Loss={final_result["qr_loss"]:.6f}')
            improvement = final_result["qr_acc"] - checkpoint_acc
            print(f'Improvement: {improvement:+.4f}%')
        print(f'{"="*80}')
        print(f'\nResults saved to: {csv_file}')


if __name__ == '__main__':
    main()
