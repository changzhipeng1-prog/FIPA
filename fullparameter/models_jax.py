"""
JAX/Flax版本的模型定义
对应PyTorch的FedNet (CNN)
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class LightFedNet(nn.Module):
    """
    超轻量级FedNet CNN模型 (JAX/Flax版本)
    参数量 ≈ 6000，适合内存受限的联邦学习
    输出类别数通过 n_classes 参数配置（默认10，可适配10类或100类）
    """
    n_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Conv1: 3 -> 4, kernel 3x3 (最小通道数)
        x = nn.Conv(features=4, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv2: 4 -> 8, kernel 3x3
        x = nn.Conv(features=8, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))  # (batch, 8*7*7 = 392)
        
        # FC1: 392 -> 16 (大幅减少全连接层参数)
        x = nn.Dense(features=16, use_bias=False)(x)
        x = nn.relu(x)
        
        # FC2: 16 -> n_classes
        x = nn.Dense(features=self.n_classes, use_bias=False)(x)
        
        return x


class MediumFedNet20k(nn.Module):
    """
    约20K参数量的中等模型 (JAX/Flax版本)
    结构：
      - Conv1: 3 -> 6
      - MaxPool 2x2
      - Conv2: 6 -> 12
      - MaxPool 2x2
      - Flatten
      - FC1: ~ (12 * 7 * 7) -> 32  （实际尺寸由VALID卷积与池化决定，约2万参数）
      - FC2: 32 -> n_classes
    参数量估算（不含bias，以n_classes=10为例）：
      Conv1: 3*3*3*6 = 162
      Conv2: 3*3*6*12 = 648
      Flatten: 12*7*7 ≈ 588（按现有Light模型形状经验）
      FC1: 588*32 = 18816
      FC2: 32*n_classes = 320 (n_classes=10时) 或 3200 (n_classes=100时)
      总计≈ 162 + 648 + 18816 + (32*n_classes) ≈ 19946 + 32*(n_classes-10)（~2万）
    输出类别数通过 n_classes 参数配置（默认10，可适配10类或100类）
    """
    n_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Conv1: 3 -> 6
        x = nn.Conv(features=6, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv2: 6 -> 12
        x = nn.Conv(features=12, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # FC1: -> 32
        x = nn.Dense(features=32, use_bias=False)(x)
        x = nn.relu(x)

        # FC2: 32 -> n_classes
        x = nn.Dense(features=self.n_classes, use_bias=False)(x)

        return x

 
class LargeFedNet200k(nn.Module):
    """
    约200K参数量的较大模型 (JAX/Flax版本)

    结构:
      - Conv1: 3 -> 12, padding='SAME'
      - MaxPool 2x2
      - Conv2: 12 -> 24, padding='SAME'
      - MaxPool 2x2
      - Conv3: 24 -> 48, padding='SAME'
      - Flatten -> Dense(61) -> Dense(n_classes)

    参数量估算（不含 bias，以n_classes=10为例）:
      Conv1: 3*3*3*12 = 324
      Conv2: 3*3*12*24 = 2,592
      Conv3: 3*3*24*48 = 10,368
      Flatten: 48 * 8 * 8 = 3,072
      FC1: 3,072 * 61 = 187,392
      FC2: 61 * n_classes = 610 (n_classes=10时) 或 6100 (n_classes=100时)
      总计 ≈ 201,286 + 61*(n_classes-10) (~200K)
    输出类别数通过 n_classes 参数配置（默认10，可适配10类或100类）
    """
    n_classes: int = 10

    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Conv(features=12, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=24, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=48, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=61, use_bias=False)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.n_classes, use_bias=False)(x)

        return x

 
 
def create_model(n_classes=10, model_size='light'):
    """
    创建轻量模型 LightFedNet（可适配CIFAR-10/CIFAR-100等数据集）
    参数:
        n_classes: 分类类别数（默认10，可设置为10用于CIFAR-10，或100用于CIFAR-100）
        model_size: 保留此参数以兼容旧接口，忽略非'light'取值
    返回:
        model: LightFedNet 实例
    """
    return LightFedNet(n_classes=n_classes)


def create_medium_model_20k(n_classes: int = 10) -> MediumFedNet20k:
    """
    工厂方法：返回约20K参数量的中等模型（可适配CIFAR-10/CIFAR-100等数据集）
    参数:
        n_classes: 分类类别数（默认10，可设置为10用于CIFAR-10，或100用于CIFAR-100）
    返回:
        MediumFedNet20k模型实例
    """
    return MediumFedNet20k(n_classes=n_classes)


def create_large_model_200k(n_classes: int = 10) -> LargeFedNet200k:
    """
    工厂方法：返回约200K参数量的较大模型（可适配CIFAR-10/CIFAR-100等数据集）
    参数:
        n_classes: 分类类别数（默认10，可设置为10用于CIFAR-10，或100用于CIFAR-100）
    返回:
        LargeFedNet200k模型实例
    """
    return LargeFedNet200k(n_classes=n_classes)


def init_model(rng, model, input_shape=(1, 32, 32, 3)):
    """
    初始化模型参数
    
    参数:
        rng: JAX随机数生成器
        model: Flax模型
        input_shape: 输入形状 (batch, height, width, channels)
    
    返回:
        params: 模型参数
    """
    dummy_input = jnp.ones(input_shape)
    params = model.init(rng, dummy_input, training=False)
    return params


def count_parameters(params) -> int:
    """统计JAX/Flax参数树中的元素总数。"""
    leaves = jax.tree_util.tree_leaves(params)
    return sum(int(leaf.size) for leaf in leaves)


# ============================================================================
# ResNet18 Implementation (aligned with FedRCL's ResNet18_base)
# ============================================================================

class GroupNorm(nn.Module):
    """Group Normalization (groups=2, as in FedRCL)"""
    num_groups: int = 2
    num_channels: int = 64
    
    @nn.compact
    def __call__(self, x):
        # Flax doesn't have GroupNorm built-in, so we implement it
        # x shape: (batch, height, width, channels)
        batch, h, w, channels = x.shape
        assert channels % self.num_groups == 0
        
        # Reshape to (batch, h, w, num_groups, channels_per_group)
        channels_per_group = channels // self.num_groups
        x = x.reshape(batch, h, w, self.num_groups, channels_per_group)
        
        # Compute mean and variance over (h, w, channels_per_group)
        mean = jnp.mean(x, axis=(1, 2, 4), keepdims=True)
        var = jnp.var(x, axis=(1, 2, 4), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-5)
        
        # Reshape back
        x = x.reshape(batch, h, w, channels)
        
        # Apply scale and shift (learnable parameters)
        scale = self.param('scale', nn.initializers.ones, (channels,))
        bias = self.param('bias', nn.initializers.zeros, (channels,))
        
        return x * scale + bias


class BasicBlock(nn.Module):
    """Basic ResNet block (aligned with FedRCL)"""
    expansion: int = 1
    planes: int = 64
    stride: int = 1
    use_bn_layer: bool = False  # FedRCL uses GroupNorm when False
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # First conv
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        if self.use_bn_layer:
            out = nn.BatchNorm(use_running_average=not training)(out)
        else:
            # Use GroupNorm (groups=2)
            out = GroupNorm(num_groups=2, num_channels=self.planes)(out)
        
        out = nn.relu(out)
        
        # Second conv
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(out)
        
        if self.use_bn_layer:
            out = nn.BatchNorm(use_running_average=not training)(out)
        else:
            out = GroupNorm(num_groups=2, num_channels=self.planes)(out)
        
        # Downsample if needed
        if self.stride != 1 or x.shape[-1] != self.planes * self.expansion:
            downsample = nn.Conv(
                features=self.planes * self.expansion,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=nn.initializers.kaiming_normal()
            )(x)
            if self.use_bn_layer:
                downsample = nn.BatchNorm(use_running_average=not training)(downsample)
            else:
                downsample = GroupNorm(num_groups=2, num_channels=self.planes * self.expansion)(downsample)
        else:
            downsample = x
        
        out = out + downsample
        out = nn.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet18 implementation aligned with FedRCL's ResNet18_base
    - Uses GroupNorm (groups=2) instead of BatchNorm (use_bn_layer=False)
    - 4 layers: [2, 2, 2, 2] blocks
    - Last feature dim: 512
    - 输出类别数通过 n_classes 参数配置（默认100，可适配10类或100类）
    """
    n_classes: int = 100
    use_bn_layer: bool = False
    last_feature_dim: int = 512
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial conv layer
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        if self.use_bn_layer:
            x = nn.BatchNorm(use_running_average=not training)(x)
        else:
            x = GroupNorm(num_groups=2, num_channels=64)(x)
        
        x = nn.relu(x)
        
        # Layer 1: 64 channels, 2 blocks
        in_planes = 64
        for i in range(2):
            x = BasicBlock(planes=64, stride=1 if i > 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Layer 2: 128 channels, 2 blocks
        for i in range(2):
            x = BasicBlock(planes=128, stride=2 if i == 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Layer 3: 256 channels, 2 blocks
        for i in range(2):
            x = BasicBlock(planes=256, stride=2 if i == 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Layer 4: last_feature_dim channels, 2 blocks
        for i in range(2):
            x = BasicBlock(planes=self.last_feature_dim, stride=2 if i == 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Global average pooling
        # x shape: (batch, h, w, channels)
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        
        # Fully connected layer
        x = nn.Dense(features=self.n_classes, use_bias=True)(x)
        
        return x


def create_resnet18(n_classes: int = 100, use_bn_layer: bool = False, last_feature_dim: int = 512) -> ResNet18:
    """
    创建ResNet18模型（与FedRCL的ResNet18_base对齐，可适配CIFAR-10/CIFAR-100等数据集）
    
    参数:
        n_classes: 分类类别数（默认100，可设置为10用于CIFAR-10，或100用于CIFAR-100）
        use_bn_layer: 是否使用BatchNorm（False时使用GroupNorm）
        last_feature_dim: 最后一层特征维度（默认512）
    
    返回:
        ResNet18模型实例
    """
    return ResNet18(n_classes=n_classes, use_bn_layer=use_bn_layer, last_feature_dim=last_feature_dim)


# ============================================================================
# ResNet-20 Implementation
# ============================================================================

class ResNet20(nn.Module):
    """
    ResNet-20 implementation (适用于CIFAR-10/CIFAR-100等数据集)
    - Structure: [3, 3, 3] blocks in 3 stages
    - Channels: [16, 32, 64]
    - ~270k parameters
    - Uses GroupNorm (groups=2) instead of BatchNorm (use_bn_layer=False)
    - 输出类别数通过 n_classes 参数配置（默认100，可适配10类或100类）
    """
    n_classes: int = 100
    use_bn_layer: bool = False
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial conv layer: 3 -> 16, kernel 3x3, no stride
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=nn.initializers.kaiming_normal()
        )(x)
        
        if self.use_bn_layer:
            x = nn.BatchNorm(use_running_average=not training)(x)
        else:
            x = GroupNorm(num_groups=2, num_channels=16)(x)
        
        x = nn.relu(x)
        
        # Stage 1: 16 channels, 3 blocks
        for i in range(3):
            x = BasicBlock(planes=16, stride=1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Stage 2: 32 channels, 3 blocks (stride=2 for first block)
        for i in range(3):
            x = BasicBlock(planes=32, stride=2 if i == 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Stage 3: 64 channels, 3 blocks (stride=2 for first block)
        for i in range(3):
            x = BasicBlock(planes=64, stride=2 if i == 0 else 1, use_bn_layer=self.use_bn_layer)(x, training=training)
        
        # Global average pooling
        # x shape: (batch, h, w, channels)
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling
        
        # Fully connected layer: 64 -> n_classes
        x = nn.Dense(features=self.n_classes, use_bias=True)(x)
        
        return x


def create_resnet20(n_classes: int = 100, use_bn_layer: bool = False) -> ResNet20:
    """
    创建ResNet-20模型（可适配CIFAR-10/CIFAR-100等数据集）
    
    参数:
        n_classes: 分类类别数（默认100，可设置为10用于CIFAR-10，或100用于CIFAR-100）
        use_bn_layer: 是否使用BatchNorm（False时使用GroupNorm）
    
    返回:
        ResNet20模型实例
    """
    return ResNet20(n_classes=n_classes, use_bn_layer=use_bn_layer)

