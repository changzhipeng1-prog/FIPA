"""
JAX版本的训练函数
"""
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Tuple
import numpy as np


def create_train_state(rng, model, learning_rate, input_shape):
    """创建训练状态"""
    params = model.init(rng, jnp.ones(input_shape), training=False)
    tx = optax.sgd(learning_rate, momentum=0.9)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


@jax.jit
def data_augmentation(rng, images):
    """
    数据增强：随机裁剪和水平翻转（JIT编译以加速）
    """
    batch_size = images.shape[0]
    # 随机裁剪 (padding=4)
    padded = jnp.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='edge')
    
    # 随机选择裁剪位置
    crop_rngs = jax.random.split(rng, batch_size)
    
    def crop_single(rng_i, img_i):
        h_offset = jax.random.randint(rng_i, (), 0, 9)
        w_offset = jax.random.randint(jax.random.fold_in(rng_i, 1), (), 0, 9)
        return jax.lax.dynamic_slice(img_i, (h_offset, w_offset, 0), (32, 32, 3))
    
    images = jax.vmap(crop_single)(crop_rngs, padded)
    
    # 随机水平翻转
    flip_rng = jax.random.fold_in(rng, 1)
    flip_mask = jax.random.uniform(flip_rng, (batch_size,)) > 0.5
    images = jnp.where(flip_mask[:, None, None, None],
                       images[:, :, ::-1, :],  # 翻转
                       images)
    
    return images


@jax.jit
def train_step(state, images, labels):
    """单个训练步骤（数据增强在外部完成）"""
    def loss_fn(params):
        logits = state.apply_fn(params, images, training=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss


def train_local_model(state, train_data, num_epochs, batch_size, rng, verbose_timing=False):
    """
    训练本地模型
    
    参数:
        state: 训练状态
        train_data: (X, y) 训练数据
        num_epochs: 训练轮数
        batch_size: 批次大小
        rng: 随机数生成器
        verbose_timing: 是否输出详细计时信息
    
    返回:
        state: 更新后的训练状态
        losses: 每个epoch的平均损失
        timing_info: 计时信息字典（如果verbose_timing=True）
    """
    import time
    X_train, y_train = train_data
    n_samples = len(X_train)
    
    losses = []
    timing_info = {
        'shuffle_time': [],
        'augment_time': [],
        'train_step_time': [],
        'total_time': []
    }
    
    total_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 打乱数据
        shuffle_start = time.time()
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        shuffle_time = time.time() - shuffle_start
        timing_info['shuffle_time'].append(shuffle_time)
        
        epoch_losses = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # 批次训练
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # 数据增强（在JIT外部，避免重新编译）
            augment_start = time.time()
            rng, aug_rng = jax.random.split(rng)
            batch_X_aug = data_augmentation(aug_rng, batch_X)
            # 同步等待数据增强完成
            batch_X_aug.block_until_ready()
            augment_time = time.time() - augment_start
            timing_info['augment_time'].append(augment_time)
            
            # JIT编译的训练步骤（不再包含数据增强）
            train_start = time.time()
            state, loss = train_step(state, batch_X_aug, batch_y)
            # 同步等待训练完成
            loss.block_until_ready()
            train_time = time.time() - train_start
            timing_info['train_step_time'].append(train_time)
            
            epoch_losses.append(loss)
        
        epoch_time = time.time() - epoch_start
        timing_info['total_time'].append(epoch_time)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if verbose_timing:
                avg_shuffle = np.mean(timing_info['shuffle_time'][-n_batches:]) if timing_info['shuffle_time'] else 0
                avg_augment = np.mean(timing_info['augment_time'][-n_batches:])
                avg_train = np.mean(timing_info['train_step_time'][-n_batches:])
                print(f"  Epoch {epoch}: Loss {avg_loss:.4f} | "
                      f"Shuffle: {avg_shuffle*1000:.1f}ms | "
                      f"Augment: {avg_augment*1000:.1f}ms | "
                      f"Train: {avg_train*1000:.1f}ms | "
                      f"Total: {epoch_time:.2f}s")
            else:
                print(f"  Epoch {epoch}: Loss {avg_loss:.4f}")
    
    total_time = time.time() - total_start
    timing_info['grand_total'] = total_time
    
    return state, losses, timing_info


def _make_eval_step(apply_fn):
    """Create a JIT-compiled evaluation step function with apply_fn in closure"""
    @jax.jit
    def _eval_step(params, batch_x, batch_y):
        """JIT-compiled evaluation step"""
        logits = apply_fn(params, batch_x, training=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
        predictions = jnp.argmax(logits, axis=-1)
        correct = jnp.sum(predictions == batch_y)
        return loss, correct
    return _eval_step


def evaluate_model(params, apply_fn, test_data, batch_size=1024, device='gpu'):
    """
    评估模型
    
    参数:
        params: 模型参数
        apply_fn: 模型的apply函数
        test_data: (X, y) 测试数据
    
    返回:
        accuracy: 准确率
        loss: 损失
    """
    X_test, y_test = test_data
    num_samples = X_test.shape[0]
    batch_size = max(1, int(batch_size))

    if device == 'cpu':
        target_devices = jax.devices('cpu')
    else:
        target_devices = jax.devices('gpu')
        if not target_devices:
            target_devices = jax.devices('cpu')
    target_device = target_devices[0]

    params_device = jax.device_put(params, target_device)

    # Create JIT-compiled evaluation step with apply_fn in closure
    _eval_step = _make_eval_step(apply_fn)

    total_loss = 0.0
    total_correct = 0

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_x = X_test[start:end]
        batch_y = y_test[start:end]

        batch_x = jax.device_put(batch_x, target_device)
        batch_y = jax.device_put(batch_y, target_device)

        # Use JIT-compiled evaluation step
        loss, correct = _eval_step(params_device, batch_x, batch_y)

        total_loss += float(loss) * (end - start)
        total_correct += int(correct)

    accuracy = (total_correct / num_samples) * 100.0
    loss = total_loss / num_samples

    return jnp.asarray(accuracy), jnp.asarray(loss)


def params_to_vector(params):
    """将参数字典展平为一维向量"""
    flat_params = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([p.flatten() for p in flat_params])


def vector_to_params(vector, params_template):
    """将一维向量重组为参数字典"""
    # 获取参数树结构
    flat_template = jax.tree_util.tree_leaves(params_template)
    treedef = jax.tree_util.tree_structure(params_template)
    
    # 切分向量并重组
    offset = 0
    new_params = []
    for param in flat_template:
        size = param.size
        shape = param.shape
        param_vector = vector[offset:offset+size]
        new_params.append(param_vector.reshape(shape))
        offset += size
    
    # 重建参数树
    return jax.tree_util.tree_unflatten(treedef, new_params)


def params_to_subvector(params, subtree_filter):
    """
    遍历 params pytree，只抽取满足 subtree_filter(path, leaf) == True 的叶子，
    将这些叶子展平并按顺序拼接成一维向量 flat_sub。
    
    参数:
        params: flax params pytree
        subtree_filter: 回调函数 (path, leaf) -> bool，path 是一个 tuple，例如 ('params', 'Dense_0', 'kernel')
    
    返回:
        flat_sub: jnp.ndarray, shape (d_S,)
        meta_info: 用于之后 subvector_to_params 重构子网 S 的元信息
                   包含每个叶子的 path、shape 以及在 flat_sub 中的起止索引。
    """
    # 获取所有路径和叶子节点
    # tree_flatten_with_path 返回 (paths_leaves, treedef)，其中 paths_leaves 是 (path, leaf) 元组列表
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    
    # 筛选出满足条件的路径和叶子
    selected_paths = []
    selected_leaves = []
    selected_sizes = []
    selected_shapes = []
    
    for path, leaf in paths_leaves:
        if subtree_filter(path, leaf):
            selected_paths.append(path)
            selected_leaves.append(leaf)
            selected_sizes.append(leaf.size)
            selected_shapes.append(leaf.shape)
    
    if not selected_leaves:
        # 如果没有选中任何参数，返回空向量
        return jnp.array([], dtype=jnp.float32), {
            'original_structure': jax.tree_util.tree_structure(params),
            'selected_paths': [],
            'selected_sizes': [],
            'selected_shapes': [],
        }
    
    # 展平选中的叶子节点并拼接
    flat_parts = [leaf.flatten() for leaf in selected_leaves]
    flat_sub = jnp.concatenate(flat_parts)
    
    meta_info = {
        'original_structure': jax.tree_util.tree_structure(params),
        'selected_paths': selected_paths,
        'selected_sizes': selected_sizes,
        'selected_shapes': selected_shapes,
    }
    
    return flat_sub, meta_info


def subvector_to_params(flat_sub, params_template, meta_info):
    """
    用 flat_sub 重构 params_template 中对应子网 S 的叶子，其余叶子保持不变。
    
    参数:
        flat_sub: jnp.ndarray, shape (d_S,)
        params_template: 原始的 params pytree，用于提供结构和非子网部分
        meta_info: 上面 params_to_subvector 返回的元信息
    
    返回:
        new_params: 完整的 params pytree
    """
    if not meta_info['selected_paths']:
        # 如果没有选中的参数，直接返回模板
        return params_template
    
    # 从 flat_sub 中切分出各个参数
    offset = 0
    selected_leaves_new = []
    for size, shape in zip(meta_info['selected_sizes'], meta_info['selected_shapes']):
        param_vector = flat_sub[offset:offset + size]
        selected_leaves_new.append(param_vector.reshape(shape))
        offset += size
    
    # 遍历整个参数树并应用更新
    paths_leaves, _ = jax.tree_util.tree_flatten_with_path(params_template)
    updated_leaves = []
    
    for path, leaf in paths_leaves:
        if path in meta_info['selected_paths']:
            idx = meta_info['selected_paths'].index(path)
            updated_leaves.append(selected_leaves_new[idx])
        else:
            updated_leaves.append(leaf)
    
    treedef = jax.tree_util.tree_structure(params_template)
    return jax.tree_util.tree_unflatten(treedef, updated_leaves)
