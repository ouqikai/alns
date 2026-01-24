import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# ================= 配置区域 =================
TARGET_SEED = 2025
TARGET_SCENE = 2  # 你想对比的场景

OUTPUT_DIR = "outputs"


# ===========================================

def plot_multi_method_convergence(seed, scene, output_dir):
    search_pattern = os.path.join(output_dir, f"converge_*_seed{seed}_scene{scene}.csv")
    files = glob.glob(search_pattern)

    if not files:
        print(f"未找到匹配的文件: {search_pattern}")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6), dpi=150)

    # 精心设计的配色和线型（重点突出 Proposed 和 TruckOnly 的对比）
    colors = {
        'G3': 'blue',  # 主算法 (混合)
        'TRUCKONLY': 'purple',  # 纯卡车对照组
        'G2': 'orange',  # 弱 ALNS
        'G1': 'red',  # 贪婪 (Greedy)
        'Greedy': 'red'  # 兼容不同的命名
    }
    linestyles = {
        'G3': '-',
        'TRUCKONLY': '--',
        'G2': '-.',
        'G1': ':',
        'Greedy': ':'
    }

    # 为了图例好看，给算法起个“学名”
    display_names = {
        'G3': 'Proposed (Truck+Drone ALNS)',
        'TruckOnly': 'Truck-Only ALNS',
        'G1': 'Greedy Baseline (Truck+Drone)',
        'Greedy': 'Greedy Baseline (Truck+Drone)'
    }

    for f in files:
        base_name = os.path.basename(f)
        match = re.search(r"converge_(.*?)_seed", base_name)
        if not match:
            continue
        method = match.group(1)

        df = pd.read_csv(f)
        if df.empty:
            continue

        # 计算 Total Cost (Distance + lambda * Late)
        # 假设你的迟到惩罚系数是 50
        df['Total_Cost'] = df['best_cost_dist'] + 50.0 * df['best_total_late']

        color = colors.get(method, 'black')
        linestyle = linestyles.get(method, '-')
        label_name = display_names.get(method, method)

        # 区分是“迭代算法”还是“静态基线”
        if len(df) > 1:
            # ALNS 类算法（包括 G3 和 TruckOnly）
            plt.plot(df['iter'], df['Total_Cost'], label=label_name,
                     color=color, linestyle=linestyle, linewidth=2.5 if method == 'G3' else 2.0)

            # 标注最终收敛值
            final_iter = df['iter'].iloc[-1]
            final_cost = df['Total_Cost'].iloc[-1]
            plt.scatter([final_iter], [final_cost], color=color, s=40)
            plt.text(final_iter, final_cost, f"{final_cost:.1f}", color=color, fontsize=9, va='bottom')
        else:
            # Greedy 类静态算法
            single_val = df['Total_Cost'].iloc[0]
            plt.axhline(y=single_val, label=label_name,
                        color=color, linestyle=linestyle, linewidth=2)
            plt.text(0, single_val, f"{single_val:.1f}", color=color, fontsize=9, va='bottom')

    # 图表装饰
    plt.title(f"Algorithm Convergence Comparison (Scene {scene}, Seed {seed})", fontsize=15, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Total Cost (Distance + Penalty)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best', frameon=True, shadow=True)

    # 自动调整 y 轴范围，避免 Greedy 数值太大导致曲线被压扁
    plt.tight_layout()

    # 保存
    out_img = os.path.join(output_dir, f"compare_converge_seed{seed}_scene{scene}.png")
    plt.savefig(out_img)
    print(f"✅ 对比图已成功生成并保存至: {out_img}")
    plt.show()


if __name__ == "__main__":
    plot_multi_method_convergence(TARGET_SEED, TARGET_SCENE, OUTPUT_DIR)