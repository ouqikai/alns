# -*- coding: utf-8 -*-
"""\
画收敛曲线（A）：best_cost_dist vs iteration。

用法示例：
  python plot_converge.py outputs/converge_G3_seed2023_scene1.csv

输出：同目录下生成 .png
"""

import os
import csv
import sys

import matplotlib.pyplot as plt


def read_converge_csv(path: str):
    its = []
    best_cost = []
    best_late = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            its.append(int(float(row.get("iter", 0))))
            best_cost.append(float(row.get("best_cost_dist", "nan")))
            best_late.append(float(row.get("best_total_late", "nan")))
    return its, best_cost, best_late


def main():
    if len(sys.argv) < 2:
        print("用法：python plot_converge.py <converge_csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    its, best_cost, best_late = read_converge_csv(csv_path)

    plt.figure()
    plt.plot(its, best_cost)
    plt.xlabel("iteration")
    plt.ylabel("best_cost_dist (truck + alpha*drone)")

    # 标题里带上文件名，方便你批量跑图后对照
    plt.title(os.path.basename(csv_path))

    out_png = os.path.splitext(csv_path)[0] + ".png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("[OK] saved:", out_png)


if __name__ == "__main__":
    main()
