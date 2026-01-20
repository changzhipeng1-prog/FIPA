import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
import pandas as pd
import pickle
import os
import glob
import argparse

# 启用 JAX 64位精度
jax.config.update("jax_enable_x64", True)

# ==========================================
# 1. 全局绘图样式配置 (Nature 风格)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 18,
    'font.weight': 'bold',
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'axes.titlesize': 22,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 30,
    'lines.linewidth': 3,
    'grid.alpha': 0.3,
    'figure.titlesize': 22,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
})

# 颜色定义（Nature子刊配色，深色版本，对比更明显）
C_EXACT = '#000000'        # 黑色
C_FEDAVG_20 = '#D68910'     # 深橙色
C_FEDAVG_1000 = '#005F8F'   # 深蓝色
C_FULLRANK = '#006B4D'      # 深绿色
C_LOWRANK = '#A0526D'       # 深紫红色

# ==========================================
# 2. 共享核心函数 (模型与工具)
# ==========================================

def mlp_network(activation):
    """Multi-layer network compatible with the saved parameter structure.
    
    params = [{"W": w, "b": b}, ...]
    For each layer l: W has shape (out_dim, in_dim), b has shape (out_dim,)
    """
    def model(params, x):
        # 处理输入：确保x是2D (batch_size, input_dim)
        if x.ndim == 0:
            h = jnp.array([[x]])
        elif x.ndim == 1:
            h = x.reshape(1, -1)
        else:
            h = x
        
        # Forward through all layers except the last one (with activation)
        for i, lyr in enumerate(params[:-1]):
            z = h @ lyr["W"].T + lyr["b"]
            h = activation(z)
        # Last layer (no activation for regression)
        w_last, b_last = params[-1]["W"], params[-1]["b"]
        out = h @ w_last.T + b_last
        # 返回标量或向量
        if out.size == 1:
            return out.reshape(-1)[0]
        return out.reshape(-1)
    return model

def load_model_params(params_file):
    """从 pickle 文件加载模型参数"""
    if not os.path.exists(params_file):
        return None
    with open(params_file, 'rb') as f:
        params_numpy = pickle.load(f)
    
    def to_jax(x):
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x
    return jax.tree_util.tree_map(to_jax, params_numpy)

def find_result_dir(method, n_clients, n_freq, local_epochs=20, step_size=0.5, top_k=20):
    """查找结果目录
    
    Args:
        method: 'fedavg', 'fullrank', 'lowrank'
        n_clients: 客户端数量
        n_freq: 频率参数
        local_epochs: 本地训练轮数
        step_size: 步长（仅用于fullrank和lowrank）
        top_k: top k值（仅用于lowrank）
    """
    # 方法名映射（目录名和结果名前缀）
    method_dir_map = {'fedavg': 'FedAvg', 'fullrank': 'fullrank', 'lowrank': 'lowrank'}
    method_name_map = {'fedavg': 'fedavg', 'fullrank': 'fullrank', 'lowrank': 'lowrank'}
    
    method_dir = method_dir_map.get(method)
    method_name = method_name_map.get(method)
    if not method_dir or not method_name:
        return None
    
    base_dir = os.path.join(method_dir, f"Results_{method_name}_N{n_clients}_nf{n_freq}")
    if not os.path.exists(base_dir):
        return None
    
    if method == 'fedavg':
        # FedAvg可能有多个epochs配置，优先选择local_epochs=20
        pattern = os.path.join(base_dir, f"{method_name}_tb-30s_mr-*_epochs{local_epochs}")
        matching_dirs = sorted(glob.glob(pattern))
        if not matching_dirs:
            # 如果没有找到，尝试找任意epochs的
            pattern = os.path.join(base_dir, f"{method_name}_tb-30s_mr-*_epochs*")
            matching_dirs = sorted(glob.glob(pattern))
    elif method == 'fullrank':
        pattern = os.path.join(base_dir, f"{method_name}_tb-30s_mr-*_epochs{local_epochs}_step{step_size}")
        matching_dirs = sorted(glob.glob(pattern))
    elif method == 'lowrank':
        pattern = os.path.join(base_dir, f"{method_name}_tb-30s_mr-*_epochs{local_epochs}_step{step_size}_k{top_k}")
        matching_dirs = sorted(glob.glob(pattern))
    else:
        return None
    
    if not matching_dirs:
        return None
    return matching_dirs[-1]

def load_loss_data(result_dir):
    """加载损失数据和时间数据
    
    根据global_test.csv的行数，将0-30秒均匀分配时间点
    
    Returns:
        time_array: 均匀分配的时间数组（0-30秒）
        test_loss: 测试损失数组
        train_loss: 训练损失数组
    """
    global_test_file = os.path.join(result_dir, 'global_test.csv')
    
    if not os.path.exists(global_test_file):
        return None, None, None
    
    # 读取损失数据
    df_loss = pd.read_csv(global_test_file)
    
    # 获取数据行数（排除header）
    n_rows = len(df_loss)
    
    # 将0-30秒均匀分成n_rows份
    time_array = np.linspace(0.0, 30.0, n_rows)
    
    return time_array, df_loss['global_test_mse'].values, df_loss['global_train_mse'].values

# ==========================================
# 3. 绘图模块：第一行 (Loss vs Time for Different Client Configurations)
# ==========================================

def plot_loss_vs_time_clients(ax, n_clients, n_freq, title):
    """绘制三种方法的测试损失随时间变化，针对特定客户端配置
    
    FedAvg显示一条曲线
    FullRank和LowRank各显示一条曲线：τ=20
    """
    print(f"[Row 1] Processing {title} (N={n_clients}, n_freq={n_freq})")
    
    step_size = 0.5
    top_k = 20
    
    # 查找结果目录
    # FedAvg: 只使用local_epochs=1000
    fedavg_dir_1000 = find_result_dir('fedavg', n_clients, n_freq, local_epochs=1000)
    # FullRank和LowRank: 使用local_epochs=20
    fullrank_dir = find_result_dir('fullrank', n_clients, n_freq, local_epochs=20, step_size=step_size)
    lowrank_dir = find_result_dir('lowrank', n_clients, n_freq, local_epochs=20, step_size=step_size, top_k=top_k)
    
    # 加载数据（只加载test loss）
    data_list = []
    
    if fedavg_dir_1000:
        time_fa1000, test_fa1000, _ = load_loss_data(fedavg_dir_1000)
        if time_fa1000 is not None:
            # FedAvg用点线，统一粗线宽
            data_list.append(('FedAvg', time_fa1000, test_fa1000, C_FEDAVG_1000, ':', 'o', 3.5))
            print(f"  FedAvg: {len(time_fa1000)} points from {os.path.basename(fedavg_dir_1000)}")
    
    if fullrank_dir:
        time_fr, test_fr, _ = load_loss_data(fullrank_dir)
        if time_fr is not None:
            # 我们的方法用实线，统一粗线宽
            data_list.append(('FIPA', time_fr, test_fr, C_FULLRANK, '-', 's', 3.5))
            print(f"  FIPA: {len(time_fr)} points from {os.path.basename(fullrank_dir)}")
    
    if lowrank_dir:
        time_lr, test_lr, _ = load_loss_data(lowrank_dir)
        if time_lr is not None:
            # 我们的方法用虚线，统一粗线宽
            data_list.append(('FIPA(r=20)', time_lr, test_lr, C_LOWRANK, '--', '^', 3.5))
            print(f"  FIPA(r=20): {len(time_lr)} points from {os.path.basename(lowrank_dir)}")
    
    if not data_list:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # 绘制测试损失（使用不同的线型和线宽，不显示marker点）
    for item in data_list:
        if len(item) == 7:  # 包含线宽
            label, time, test, color, linestyle, marker, linewidth = item
        else:  # 兼容旧格式
            label, time, test, color, linestyle, marker = item
            linewidth = 3
        ax.semilogy(time, test, label=label, color=color, linestyle=linestyle, 
                   linewidth=linewidth)
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=20)
    if n_clients == 2:
        ax.set_ylabel('Test Loss (Log Scale)', fontweight='bold', fontsize=20)
    # 标题使用M表示：M=2, M=4, M=8
    title_with_M = title.replace('2 Clients', '$\sin(2\pi x)$').replace('4 Clients', '$\sin(4\pi x)$').replace('8 Clients', '$\sin(8\pi x)$')
    ax.set_title(title_with_M, fontweight='bold', fontsize=22)
    ax.set_xlim(0, 30)
    ax.grid(False)
    
    # 设置y轴刻度格式：使用简单数字（-1, -2）而不是科学计数法（10^{-1}）
    def log_formatter(x, pos):
        """将10的幂次转换为简单数字"""
        if x <= 0:
            return ''
        exp = int(np.log10(x))
        return f'{exp}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    # 放大刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # 不在这里显示图例，将在外部统一显示

# ==========================================
# 4. 绘图模块：第二行 (1D Fitting for Different Client Configurations)
# ==========================================

def u_exact_1d(x, n_freq=1):
    return jnp.sin(n_freq * jnp.pi * x)

def plot_fitting_1d_clients(ax, edges, n_clients, n_freq, title):
    """绘制1D函数拟合结果，展示不同客户端划分"""
    print(f"[Row 2] Processing {title} (N={n_clients}, n_freq={n_freq})")
    
    # 查找结果目录
    # FedAvg使用local_epochs=1000，FIPA使用local_epochs=20
    step_size = 0.5
    top_k = 20
    
    fedavg_dir = find_result_dir('fedavg', n_clients, n_freq, local_epochs=1000)
    fullrank_dir = find_result_dir('fullrank', n_clients, n_freq, local_epochs=20, step_size=step_size)
    lowrank_dir = find_result_dir('lowrank', n_clients, n_freq, local_epochs=20, step_size=step_size, top_k=top_k)
    
    if not fedavg_dir or not fullrank_dir or not lowrank_dir:
        ax.text(0.5, 0.5, f'No data for N={n_clients}', ha='center', va='center', transform=ax.transAxes)
        return
    
    # 加载模型参数
    params_fa = load_model_params(os.path.join(fedavg_dir, 'final_params.pkl'))
    params_fr = load_model_params(os.path.join(fullrank_dir, 'final_params.pkl'))
    params_lr = load_model_params(os.path.join(lowrank_dir, 'final_params.pkl'))
    
    if params_fa is None or params_fr is None or params_lr is None:
        ax.text(0.5, 0.5, 'Missing params', ha='center', va='center', transform=ax.transAxes)
        return
    
    model = mlp_network(jnp.tanh)
    n_points = 300
    x_plot = jnp.linspace(0.0, 1.0, n_points)
    y_exact = u_exact_1d(x_plot, n_freq=n_freq)
    
    # 使用vmap批量处理
    v_model = jax.vmap(lambda p, x: model(p, x.reshape(1, 1)), in_axes=(None, 0))
    y_fa = v_model(params_fa, x_plot)
    y_fr = v_model(params_fr, x_plot)
    y_lr = v_model(params_lr, x_plot)
    x_np = np.array(x_plot)
    
    # 绘图（使用与第一行相同的线型、颜色和线宽，确保一致性）
    ax.plot(x_np, np.array(y_exact), label='Exact', color=C_EXACT, linewidth=3, linestyle='-')
    ax.plot(x_np, np.array(y_fa), label='FedAvg', color=C_FEDAVG_1000, linewidth=3.5, alpha=1.0, linestyle=':', marker='o', markersize=4, markevery=max(1, len(x_np)//30))
    ax.plot(x_np, np.array(y_fr), label='FIPA', color=C_FULLRANK, linewidth=3.5, alpha=1.0, linestyle='-', marker='s', markersize=4, markevery=max(1, len(x_np)//30))
    ax.plot(x_np, np.array(y_lr), label='FIPA(r=20)', color=C_LOWRANK, linewidth=3.5, alpha=1.0, linestyle='--', marker='^', markersize=4, markevery=max(1, len(x_np)//30))
    
    # Client 背景与标注
    # 使用固定的y轴范围中间位置（-1.1到1.1的中间，即0）
    y_mid = 0.0
    
    # 绘制客户端区域
    colors = plt.cm.Set3(np.linspace(0, 1, n_clients))
    # 对于M=8（最后一幅图），使用更浅的颜色
    if n_clients == 8:
        alpha_bg = 0.08  # 背景更浅
        alpha_text = 0.5  # 文字背景更浅
    else:
        alpha_bg = 0.15
        alpha_text = 0.7
    
    for i in range(n_clients):
        left = edges[i]
        right = edges[i + 1]
        ax.axvspan(left, right, alpha=alpha_bg, color=colors[i], zorder=0)
        center = (left + right) / 2
        ax.text(center, y_mid, f'C{i}', ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=alpha_text, edgecolor='black', linewidth=1))
    
    # 绘制客户端边界
    for i in range(1, n_clients):
        ax.axvline(x=edges[i], color='red', linestyle=':', linewidth=2, alpha=0.7, zorder=1)
    
    ax.set_xlabel('x', fontweight='bold', fontsize=20)
    if n_clients == 2:
        ax.set_ylabel('f(x)', fontweight='bold', fontsize=20)
    # 第二行不显示标题
    # ax.set_title(title, fontweight='bold', fontsize=22)
    ax.set_xlim(0, 1)
    # xticks只在clients的边界上显示
    ax.set_xticks(edges)
    # 如果xticks太多，旋转标签避免重叠
    if len(edges) > 5:
        ax.tick_params(axis='x', which='major', labelsize=16, rotation=45)
    else:
        ax.tick_params(axis='x', which='major', labelsize=18)
    # 设置y轴范围和刻度
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(False)
    # 放大刻度标签字体
    ax.tick_params(axis='y', which='major', labelsize=18)
    
    # 不在这里显示图例，将在外部统一显示

# ==========================================
# 5. 主程序
# ==========================================

def main():
    print("Starting combined figure generation...")
    fig = plt.figure(figsize=(22, 12))
    
    # 外层Grid：三行（1行图例 + 2行子图）
    outer_gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.12, 1.0, 1.0], hspace=0.4)
    
    # 图例行（最上方，居中）
    ax_legend = fig.add_subplot(outer_gs[0, 0])
    ax_legend.axis('off')
    
    # 创建统一的图例（所有出现过的线，使用Nature配色和不同线型，不显示marker）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_FEDAVG_1000, linestyle=':', linewidth=3.5, label='FedAvg'),
        Line2D([0], [0], color=C_FULLRANK, linestyle='-', linewidth=3.5, label='FIPA'),
        Line2D([0], [0], color=C_LOWRANK, linestyle='--', linewidth=3.5, label='FIPA(r=20)'),
        Line2D([0], [0], color=C_EXACT, linestyle='-', linewidth=3, label='Exact'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', framealpha=0.9, fontsize=30, ncol=4)
    
    # Row 1: 3列（不再需要图例列）
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[1], wspace=0.25)
    
    # 2 clients
    ax_2c_loss = fig.add_subplot(gs_top[0, 0])
    plot_loss_vs_time_clients(ax_2c_loss, 2, 2, '2 Clients')
    
    # 4 clients
    ax_4c_loss = fig.add_subplot(gs_top[0, 1])
    plot_loss_vs_time_clients(ax_4c_loss, 4, 4, '4 Clients')
    
    # 8 clients
    ax_8c_loss = fig.add_subplot(gs_top[0, 2])
    plot_loss_vs_time_clients(ax_8c_loss, 8, 8, '8 Clients')
    
    # 统一第一行的y轴范围
    y_min_top = min(ax_2c_loss.get_ylim()[0], ax_4c_loss.get_ylim()[0], ax_8c_loss.get_ylim()[0])
    y_max_top = max(ax_2c_loss.get_ylim()[1], ax_4c_loss.get_ylim()[1], ax_8c_loss.get_ylim()[1])
    ax_2c_loss.set_ylim(y_min_top, y_max_top)
    ax_4c_loss.set_ylim(y_min_top, y_max_top)
    ax_8c_loss.set_ylim(y_min_top, y_max_top)
    
    # Row 2: 3列（不再需要图例列）
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[2], wspace=0.25)
    
    # 2 clients
    edges_2 = [0.0, 0.5, 1.0]
    ax_2c_fit = fig.add_subplot(gs_bottom[0, 0])
    plot_fitting_1d_clients(ax_2c_fit, edges_2, 2, 2, '2 Clients')
    
    # 4 clients
    edges_4 = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax_4c_fit = fig.add_subplot(gs_bottom[0, 1])
    plot_fitting_1d_clients(ax_4c_fit, edges_4, 4, 4, '4 Clients')
    
    # 8 clients
    edges_8 = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.85, 0.9, 1.0]
    ax_8c_fit = fig.add_subplot(gs_bottom[0, 2])
    plot_fitting_1d_clients(ax_8c_fit, edges_8, 8, 8, '8 Clients')
    
    # 统一第二行的y轴范围（固定为-1.1到1.1，0.0在中间）
    ax_2c_fit.set_ylim(-1.1, 1.1)
    ax_4c_fit.set_ylim(-1.1, 1.1)
    ax_8c_fit.set_ylim(-1.1, 1.1)
    # 设置yticks只显示-1, -0.5, 0, 0.5, 1
    ax_2c_fit.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_4c_fit.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_8c_fit.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    output_filename = 'combined_all_figures_final.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved successfully: {output_filename}")

if __name__ == "__main__":
    main()
