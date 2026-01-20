"""FedAvg warmup training with checkpoint saving every 100 rounds."""

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
from train_jax import train_local_model, evaluate_model
from data_jax import load_cifar10_data


os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)  # Aligned with FedRCL (seed=0)
jax.config.update('jax_default_prng_impl', 'rbg')
jax.config.update('jax_enable_x64', False)  # Single precision (float32), aligned with PyTorch
np.random.seed(0)  # Aligned with FedRCL (seed=0)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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


def save_checkpoint(params, round_idx, checkpoint_dir, initial_lr, lr_decay):
    """Save model parameters as checkpoint with metadata."""
    checkpoint_path = Path(checkpoint_dir) / f'checkpoint_round_{round_idx}.pkl'
    checkpoint_meta = {
        'params': params,
        'round': round_idx,
        'initial_lr': initial_lr,
        'lr_decay': lr_decay,
        'current_lr': initial_lr * (lr_decay ** round_idx),  # LR at this round
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_meta, f)
    print(f'  Checkpoint saved: {checkpoint_path} (round {round_idx}, lr={checkpoint_meta["current_lr"]:.6f})')


def main():
    """Entry point for FedAvg warmup training with checkpoint saving."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        parser = argparse.ArgumentParser(description='FedAvg warmup with checkpoint saving')
        parser.add_argument('--initial_lr', type=float, default=0.1)  # Aligned with FedRCL
        parser.add_argument('--warmup_rounds', type=int, default=1000, help='FedAvg warmup rounds')
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--total_clients', type=int, default=100)  # Aligned with train_fedavg_only.py
        parser.add_argument('--client_fraction', type=float, default=0.05)  # Aligned with train_fedavg_only.py
        parser.add_argument('--local_epochs', type=int, default=5)
        parser.add_argument('--client_selection_seed', type=int, default=0)  # Aligned with train_fedavg_only.py
        parser.add_argument('--model_size', type=str, default='resnet20', choices=['light', 'medium20k', 'large200k', 'resnet20'])
        parser.add_argument('--eval_batch_size', type=int, default=128)  # Aligned with FedRCL
        parser.add_argument('--train_batch_size', type=int, default=50)  # Aligned with FedRCL
        parser.add_argument('--weight_decay', type=float, default=1e-3)  # Aligned with FedRCL
        parser.add_argument('--lr_decay', type=float, default=0.998)  # Aligned with FedRCL (per round)
        parser.add_argument('--checkpoint_dir', type=str, default='fedavg_checkpoints')
        parser.add_argument('--checkpoint_interval', type=int, default=100, help='Save checkpoint every N rounds')
        args = parser.parse_args()
        config = vars(args)
    else:
        config = None

    config = comm.bcast(config, root=0)

    initial_lr = config['initial_lr']
    warmup_rounds = config['warmup_rounds']
    alpha = config['alpha']
    total_clients = config['total_clients']
    client_fraction = config['client_fraction']
    local_epochs = int(config['local_epochs'])
    client_selection_seed = config['client_selection_seed']
    model_size = config['model_size']
    eval_batch_size = config['eval_batch_size']
    train_batch_size = config.get('train_batch_size', 50)
    weight_decay = config.get('weight_decay', 1e-3)
    lr_decay = config.get('lr_decay', 0.998)
    checkpoint_dir = config['checkpoint_dir']
    checkpoint_interval = config.get('checkpoint_interval', 100)

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
    
    if rank == 0:
        print(f'Config: model={model_size}, total_clients={total_clients}, fraction={client_fraction:.2f}, '
              f'warmup={warmup_rounds}, local_epochs={local_epochs}')
        print(f'GPU assignment: {num_gpus} ranks/GPUs serving {total_clients} logical clients (max {num_gpus} active per round)')
        print(f'Checkpoint directory: {checkpoint_dir}, interval: {checkpoint_interval} rounds')

    # Load federated datasets
    train_datasets, test_data, _ = load_cifar10_data(total_clients, alpha)
    client_samples = np.array([len(data[0]) for data in train_datasets])
    weights = client_samples / client_samples.sum()

    if rank == 0:
        for i, count in enumerate(client_samples):
            print(f'Client {i} Training examples: {count}')

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

    if rank == 0:
        global_params = init_model(jax.random.PRNGKey(0), model)  # Aligned with FedRCL (seed=0)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        global_params = None

    global_params = comm.bcast(global_params, root=0)
    comm.Barrier()

    rng = jax.random.PRNGKey(0 + rank)  # Aligned with FedRCL and train_fedavg_only.py (seed=0)
    results = []

    # FedAvg Warmup Training
    # =================================================================
    if rank == 0:
        print(f'\n=== FedAvg Warmup Training ({warmup_rounds} rounds) ===')
    
    for round_idx in range(warmup_rounds):
        # Learning rate decay (aligned with FedRCL: local_lr_decay=0.998 per round)
        current_lr = initial_lr * (lr_decay ** round_idx)
        if rank == 0:
            warmup_round_start = time.time()
        if rank == 0:
            active_indices = sample_active_clients(total_clients, client_fraction, client_selection_seed + round_idx, num_gpus)
            print(f'\n[Warmup] Round {round_idx+1}/{warmup_rounds}, Active clients: {active_indices}')
        else:
            active_indices = None
        
        active_indices = comm.bcast(active_indices, root=0)
        active_set = set(active_indices)

        assigned_cid = None
        for cid in active_indices:
            if cid % size == rank:
                assigned_cid = cid
                break

        # Local training
        if assigned_cid is not None:
            X_client, y_client = train_datasets[assigned_cid]
            rng, client_rng = jax.random.split(rng)
            # Optimizer: SGD with lr=current_lr, momentum=0.0, weight_decay=1e-3 (aligned with FedRCL)
            tx = optax.chain(
                optax.add_decayed_weights(weight_decay),  # weight decay
                optax.sgd(learning_rate=current_lr, momentum=0.0)  # momentum=0.0 (aligned with FedRCL)
            )
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=global_params,
                tx=tx,
            )
            # Use parameter dict directly instead of vector conversion (aligned with train_fedavg_only.py)
            params_before = global_params
            state, _, _ = train_local_model(state, (X_client, y_client), local_epochs, train_batch_size, client_rng)
            params_after = state.params
            # Compute delta using parameter dict directly (aligned with train_fedavg_only.py)
            delta_w_i = jax.tree_util.tree_map(lambda a, b: a - b, params_after, params_before)
            # 同步等待delta计算完成
            delta_w_i = jax.tree_util.tree_map(lambda x: x.block_until_ready(), delta_w_i)
        else:
            delta_w_i = None

        # Gather deltas with client ids
        client_delta = (assigned_cid, delta_w_i)
        all_deltas = comm.gather(client_delta, root=0)

        # Server aggregation (simple average, aligned with FedRCL and train_fedavg_only.py)
        if rank == 0:
            # Simple average instead of weighted average (aligned with FedRCL)
            num_active = len(active_indices)
            normalized_weights = {i: 1.0 / num_active for i in active_indices}
            
            delta_dict = {}
            for cid, delta in all_deltas:
                if cid is not None and delta is not None:
                    delta_dict[cid] = delta

            # Aggregate using parameter dict directly (aligned with train_fedavg_only.py)
            fedavg_delta = None
            for idx in active_indices:
                if idx in delta_dict:
                    if fedavg_delta is None:
                        fedavg_delta = jax.tree_util.tree_map(
                            lambda x: normalized_weights[idx] * x, delta_dict[idx]
                        )
                    else:
                        fedavg_delta = jax.tree_util.tree_map(
                            lambda a, b: a + normalized_weights[idx] * b,
                            fedavg_delta, delta_dict[idx]
                        )
            
            global_params = jax.tree_util.tree_map(lambda a, b: a + b, global_params, fedavg_delta)
            # 同步等待聚合完成
            global_params = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, global_params)
            test_acc, test_loss = evaluate_model(global_params, model.apply, test_data, batch_size=eval_batch_size)
            print(f'  Test Acc: {test_acc:.2f}%, Loss: {test_loss:.4f}')
            warmup_time_sec = time.time() - warmup_round_start
            print(f'  Warmup round duration: {warmup_time_sec:.2f} sec')
            
            results.append({
                'phase': 'warmup',
                'round': round_idx + 1,
                'test_acc': float(test_acc),
                'test_loss': float(test_loss),
                'phase_time_sec': float(warmup_time_sec),
            })
            
            # Save checkpoint every checkpoint_interval rounds
            if (round_idx + 1) % checkpoint_interval == 0:
                save_checkpoint(global_params, round_idx + 1, checkpoint_dir, initial_lr, lr_decay)
        
        global_params = comm.bcast(global_params, root=0)

    # Save final checkpoint
    if rank == 0:
        save_checkpoint(global_params, warmup_rounds, checkpoint_dir, initial_lr, lr_decay)
        
        # Save training results
        csv_file = Path(checkpoint_dir) / 'warmup_results.csv'
        with csv_file.open('w', newline='') as f:
            if results:
                fieldnames = [
                    'phase',
                    'round',
                    'test_acc',
                    'test_loss',
                    'phase_time_sec',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    normalized_row = {field: row.get(field, '') for field in fieldnames}
                    writer.writerow(normalized_row)
        print(f'\nWarmup results saved to: {csv_file}')
        print(f'All checkpoints saved in: {checkpoint_dir}')


if __name__ == '__main__':
    main()


