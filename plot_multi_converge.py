import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ================= 配置区域 =================
# 自动寻找 outputs 目录下最新的 compare_*.csv 文件
OUTPUT_DIR = "outputs"
TARGET_SEED = 2025  # 你想绘制的种子


# ===========================================

def plot_dynamic_cost_across_timesteps(seed, output_dir):
    # 查找所有匹配的对比文件
    search_pattern = os.path.join(output_dir, f"compare_*_seed{seed}_*.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"❌ 未找到匹配的对比文件: {search_pattern}")
        print("请确保你已经运行了 main.py 里的 RUN_COMPARE_SUITE = True")
        return

    # 获取最新的一个文件（按修改时间排序）
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ 正在读取最新的对比文件: {latest_file}")

    # 读取数据
    df = pd.read_csv(latest_file)
    if df.empty:
        print("⚠️ 数据为空")
        return

    # 过滤掉不需要的算法（例如 G0 不做规划），只保留核心对比项
    # df = df[df['method'].isin(['G3', 'TruckOnly', 'Greedy'])]

    # 设置中文字体（避免乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(9, 5.5), dpi=150)

    # 精心设计的配色和线型
    colors = {
        'Proposed': 'blue',  # G3
        'G3': 'blue',
        'TruckOnly': 'purple',  # 纯卡车
        'Greedy': 'red',  # 贪婪G1
        'G1': 'red',
        'G0': 'grey'
    }

    linestyles = {
        'Proposed': '-', 'G3': '-',
        'TruckOnly': '--',
        'Greedy': '-.', 'G1': '-.',
        'G0': ':'
    }

    markers = {
        'Proposed': 'o', 'G3': 'o',
        'TruckOnly': 's',
        'Greedy': '^', 'G1': '^',
        'G0': 'x'
    }

    display_names = {
        'Proposed': 'Proposed (Truck+Drone ALNS)',
        'G3': 'Proposed (Truck+Drone ALNS)',
        'TruckOnly': 'Truck-Only ALNS',
        'Greedy': 'Greedy Baseline',
        'G1': 'Greedy Baseline',
        'G0': 'No-Replan (Do Nothing)'
    }

    # 获取所有存在的方法
    methods = df['method'].unique()

    # 绘制每个算法的曲线
    for method in methods:
        # 按场景（时间步）排序
        sub_df = df[df['method'] == method].sort_values(by='scene')

        # 提取 X 和 Y
        x_scenes = sub_df['scene']
        y_base_cost = sub_df['base_cost']  # <--- 重点：只取不带惩罚的物理成本

        # 获取样式
        color = colors.get(method, 'black')
        linestyle = linestyles.get(method, '-')
        marker = markers.get(method, 'o')
        label_name = display_names.get(method, method)

        # 画线
        plt.plot(x_scenes, y_base_cost,
                 label=label_name,
                 color=color,
                 linestyle=linestyle,
                 marker=marker,
                 markersize=6,
                 linewidth=2.0)

        # 可选：在最后一个点标注具体数值
        final_x = x_scenes.iloc[-1]
        final_y = y_base_cost.iloc[-1]
        plt.text(final_x, final_y, f"{final_y:.1f}", color=color, fontsize=9, va='bottom', ha='left')

    # 图表装饰
    plt.title(f"Dynamic Operations Cost across Timesteps (Seed {seed})", fontsize=14, fontweight='bold')
    plt.xlabel("Timestep (Scene)", fontsize=12)
    plt.ylabel("Operational Cost without Penalty (Truck + α * Drone)", fontsize=12)

    # 强制 X 轴为整数
    plt.xticks(sorted(df['scene'].unique()))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='best', frameon=True, shadow=True)

    plt.tight_layout()

    # 保存图片
    out_img = os.path.join(output_dir, f"dynamic_cost_seed{seed}.png")
    plt.savefig(out_img)
    print(f"✅ 动态成本对比图已成功生成并保存至: {out_img}")
    plt.show()


if __name__ == "__main__":
    plot_dynamic_cost_across_timesteps(TARGET_SEED, OUTPUT_DIR)