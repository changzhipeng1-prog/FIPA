"""
JAX版本的数据加载和Dirichlet划分
注意：此模块依赖全局numpy随机种子，调用前需设置np.random.seed()
"""
import numpy as np
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_dirichlet_data(y, n_clients, alpha, num_classes, verbose=False, balanced=True):
    """
    使用Dirichlet分布划分数据
    
    参数:
        y: 标签数组
        n_clients: 客户端数量
        alpha: Dirichlet参数
        num_classes: 类别数量
        verbose: 是否打印统计信息
        balanced: 是否使用balanced版本（每个客户端固定样本数，与FedRCL对齐）
    
    返回:
        idx_batch: 每个客户端的数据索引
        net_cls_counts: 每个客户端的类别统计
    """
    N = len(y)
    net_dataidx_map = {}
    
    if balanced:
        # Balanced版本：每个客户端固定样本数（与FedRCL的cifar_dirichlet_balanced对齐）
        num_data_per_client = int(N / n_clients)
        idx_batch = [[] for _ in range(n_clients)]
        assigned_ids = []
        
        for i in range(n_clients):
            # 为每个客户端采样类别分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            
            # 计算每个样本的权重（基于类别分布）
            weights = np.zeros(N)
            for k in range(num_classes):
                idx_k = np.where(y == k)[0]
                weights[idx_k] = proportions[k]
            
            # 已分配的样本权重设为0
            weights[assigned_ids] = 0.0
            
            # 归一化权重
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # 如果所有权重为0，使用均匀分布
                weights = np.ones(N)
                weights[assigned_ids] = 0.0
                weights = weights / weights.sum()
            
            # 使用multinomial采样固定数量的样本
            # 由于numpy的multinomial不支持replacement=False，我们使用加权随机采样
            available_indices = np.where(weights > 0)[0]
            if len(available_indices) >= num_data_per_client:
                # 使用加权随机采样
                selected_indices = np.random.choice(
                    available_indices,
                    size=num_data_per_client,
                    replace=False,
                    p=weights[available_indices] / weights[available_indices].sum()
                )
            else:
                # 如果可用样本不足，使用所有可用样本
                selected_indices = available_indices
            
            idx_batch[i] = selected_indices.tolist()
            assigned_ids.extend(selected_indices.tolist())
        
        # 打乱每个客户端的数据
        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    else:
        # Unbalanced版本：原始实现
        p_client = np.zeros((n_clients, num_classes))
        
        # 为每个客户端采样类别分布
        for i in range(n_clients):
            p_client[i] = np.random.dirichlet(np.repeat(alpha, num_classes))
        
        idx_batch = [[] for _ in range(n_clients)]
        
        # 按类别分配数据
        for k in range(num_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = p_client[:, k]
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() 
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        
        # 打乱并统计
        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    
    # 统计类别分布
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    
    if verbose:
        print('Data statistics: %s' % str(net_cls_counts))
        if balanced:
            sizes = [len(net_dataidx_map[i]) for i in range(n_clients)]
            print(f'Balanced split: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}')
    
    return idx_batch, net_cls_counts


def load_cifar10_data(n_clients, alpha, verbose=False):
    """
    加载CIFAR-10数据并使用Dirichlet划分
    
    返回:
        train_datasets: 每个客户端的训练数据(numpy arrays)
        test_data: 测试数据(numpy array)
        net_cls_counts: 类别统计
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], 
                           std=[0.247, 0.243, 0.262])
    ])
    
    # 数据根目录：固定为本文件父目录下的 data/cifar10（与脚本运行目录无关）
    data_root = str((Path(__file__).resolve().parent / 'data' / 'cifar10'))
    
    # 加载数据集（不下载，直接使用现有数据）
    dataset_train_global = datasets.CIFAR10(
        data_root, train=True, download=False, transform=transform)
    dataset_test_global = datasets.CIFAR10(
        data_root, train=False, download=False, transform=transform)
    
    # 转换为numpy数组
    train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
    test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))
    
    X_train, Y_train = next(iter(train_loader))
    X_test, Y_test = next(iter(test_loader))
    
    X_train = X_train.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    Y_train = Y_train.numpy()
    X_test = X_test.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    Y_test = Y_test.numpy()
    
    # Dirichlet划分（使用balanced版本，与FedRCL对齐）
    # 🔧 注意：调用者应该在调用前设置np.random.seed()以确保可重复性
    idx_batch, net_cls_counts = get_dirichlet_data(Y_train, n_clients, alpha, 10, verbose=verbose, balanced=True)
    
    # 创建客户端数据集
    train_datasets = []
    for i, indices in enumerate(idx_batch):
        x_client = X_train[indices]
        y_client = Y_train[indices]
        train_datasets.append((x_client, y_client))
        if verbose:
            print(f"Client {i} Training examples: {len(x_client)}")
    
    return train_datasets, (X_test, Y_test), net_cls_counts


def load_cifar100_data(n_clients, alpha, verbose=False):
    """
    加载CIFAR-100数据并使用Dirichlet划分，同时在本地不存在数据时自动下载。
    
    返回:
        train_datasets: 每个客户端的训练数据(numpy arrays)
        test_data: 测试数据(numpy array)
        net_cls_counts: 类别统计
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])
    
    # 数据根目录：固定为本文件父目录下的 data/cifar100（与脚本运行目录无关）
    data_root = str((Path(__file__).resolve().parent / 'data' / 'cifar100'))
    
    # 加载数据集（不下载，直接使用现有数据）
    dataset_train_global = datasets.CIFAR100(
        data_root, train=True, download=False, transform=transform)
    dataset_test_global = datasets.CIFAR100(
        data_root, train=False, download=False, transform=transform)
    
    # 转换为numpy数组
    train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
    test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))
    
    X_train, Y_train = next(iter(train_loader))
    X_test, Y_test = next(iter(test_loader))
    
    X_train = X_train.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    Y_train = Y_train.numpy()
    X_test = X_test.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    Y_test = Y_test.numpy()
    
    # Dirichlet划分（使用balanced版本，与FedRCL对齐）
    # 🔧 注意：调用者应该在调用前设置np.random.seed()以确保可重复性
    idx_batch, net_cls_counts = get_dirichlet_data(Y_train, n_clients, alpha, 100, verbose=verbose, balanced=True)
    
    # 创建客户端数据集
    train_datasets = []
    for i, indices in enumerate(idx_batch):
        x_client = X_train[indices]
        y_client = Y_train[indices]
        train_datasets.append((x_client, y_client))
        if verbose:
            print(f"Client {i} Training examples: {len(x_client)}")
    
    return train_datasets, (X_test, Y_test), net_cls_counts


