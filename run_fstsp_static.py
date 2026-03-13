import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def solve_fstsp_static_and_plot(
        csv_path, E_roundtrip_km=10.0, truck_speed_kmh=30.0,
        truck_road_factor=1.5, drone_speed_kmh=60.0, alpha=0.3,
        lambda_late=50.0, time_limit=1800.0, mip_gap=0.0, unit_per_km=5.0
):
    print(">>> 开始读取数据并过滤基站...")
    df = pd.read_csv(csv_path)

    # 核心：彻底屏蔽所有 base 节点，FSTSP 字典里查无此物！
    df_cust = df[df['NODE_TYPE'].str.lower() == 'customer'].reset_index(drop=True)
    depot_row = df[df['NODE_TYPE'].str.lower() == 'central'].iloc[0]

    c = len(df_cust)
    nodeNum = c + 2

    corX = [0.0] * nodeNum
    corY = [0.0] * nodeNum
    due_time = [999999.0] * nodeNum
    idx2id = {}

    # 0 = 中心起点
    idx2id[0] = int(depot_row['NODE_ID'])
    corX[0] = float(depot_row['ORIG_X'])
    corY[0] = float(depot_row['ORIG_Y'])

    # 1 to c = 客户点
    for i in range(c):
        idx = i + 1
        idx2id[idx] = int(df_cust.loc[i, 'NODE_ID'])
        corX[idx] = float(df_cust.loc[i, 'ORIG_X'])
        corY[idx] = float(df_cust.loc[i, 'ORIG_Y'])
        due_time[idx] = float(df_cust.loc[i, 'EFFECTIVE_DUE']) if 'EFFECTIVE_DUE' in df_cust.columns else float(
            df_cust.loc[i, 'DUE_TIME'])

    # c+1 = 中心终点
    idx2id[c + 1] = int(depot_row['NODE_ID'])
    corX[c + 1] = float(depot_row['ORIG_X'])
    corY[c + 1] = float(depot_row['ORIG_Y'])

    N_out_lst = list(range(0, c + 1))
    N_in_lst = list(range(1, c + 2))
    C_lst = list(range(1, c + 1))

    distMatrix = np.zeros((nodeNum, nodeNum))
    costMatrix = np.zeros((nodeNum, nodeNum))

    for i in range(nodeNum):
        for j in range(nodeNum):
            if i != j:
                d_units = euclid(corX[i], corY[i], corX[j], corY[j])
                distMatrix[i][j] = d_units / unit_per_km
                costMatrix[i][j] = (distMatrix[i][j] * truck_road_factor) / truck_speed_kmh

    UAV_factor = (1.0 / drone_speed_kmh) / (truck_road_factor / truck_speed_kmh)

    P_lst = []
    for i in N_out_lst:
        for j in C_lst:
            if i != j:
                for k in N_in_lst:
                    if k != j and k != i:
                        fly_time_ij = distMatrix[i][j] / drone_speed_kmh
                        fly_time_jk = distMatrix[j][k] / drone_speed_kmh
                        if fly_time_ij + fly_time_jk <= (E_roundtrip_km / drone_speed_kmh) + 1e-6:
                            P_lst.append((i, j, k))

    m = gp.Model('FSTSP_Static')
    m.setParam('TimeLimit', time_limit)
    m.setParam('MIPGap', mip_gap)

    M = 100000.0
    X = m.addVars(nodeNum, nodeNum, vtype=GRB.BINARY, name="X")
    Y = m.addVars(nodeNum, nodeNum, nodeNum, vtype=GRB.BINARY, name="Y")
    U = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=1.0, ub=c + 2, name="U")
    T = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    TUAV = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="TUAV")
    F = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    L = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="Late")

    # FSTSP 核心约束
    for j in C_lst:
        m.addConstr(gp.quicksum(X[i, j] for i in N_out_lst if i != j) +
                    gp.quicksum(Y[i, j, k] for i in N_out_lst for k in N_in_lst if (i, j, k) in P_lst and i != j) == 1)

    m.addConstr(gp.quicksum(X[0, j] for j in N_in_lst) == 1)
    m.addConstr(gp.quicksum(X[i, c + 1] for i in N_out_lst) == 1)

    for j in C_lst:
        m.addConstr(
            gp.quicksum(X[i, j] for i in N_out_lst if i != j) == gp.quicksum(X[j, k] for k in N_in_lst if k != j))

    for i in C_lst:
        for j in N_in_lst:
            if i != j:
                m.addConstr(U[i] - U[j] + (c + 2) * X[i, j] <= c + 1)

    for i in N_out_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst and j != i) <= 1)
    for k in N_in_lst:
        m.addConstr(gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst and i != k) <= 1)

    for i in C_lst:
        for j in C_lst:
            if j != i:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(gp.quicksum(X[h, i] for h in N_out_lst if h != i) + gp.quicksum(
                            X[l, k] for l in C_lst if l != k) >= 2 * Y[i, j, k])
    for j in C_lst:
        for k in N_in_lst:
            if (0, j, k) in P_lst:
                m.addConstr(gp.quicksum(X[h, k] for h in N_out_lst if h != k) >= Y[0, j, k])

    for i in C_lst:
        m.addConstr(
            T[i] - TUAV[i] <= M * (1 - gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst)))
        m.addConstr(
            TUAV[i] - T[i] <= M * (1 - gp.quicksum(Y[i, j, k] for j in C_lst for k in N_in_lst if (i, j, k) in P_lst)))
    for k in N_in_lst:
        m.addConstr(
            T[k] - TUAV[k] <= M * (1 - gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst)))
        m.addConstr(
            TUAV[k] - T[k] <= M * (1 - gp.quicksum(Y[i, j, k] for i in N_out_lst for j in C_lst if (i, j, k) in P_lst)))

    for h in N_out_lst:
        for k in N_in_lst:
            if h != k:
                m.addConstr(T[h] - T[k] + costMatrix[h][k] <= M * (1 - X[h, k]))

    for j in C_lst:
        for i in N_out_lst:
            if i != j:
                m.addConstr(TUAV[i] - TUAV[j] + UAV_factor * costMatrix[i][j] <= M * (
                            1 - gp.quicksum(Y[i, j, k] for k in N_in_lst if (i, j, k) in P_lst)))
        for k in N_in_lst:
            if k != j:
                m.addConstr(TUAV[j] - TUAV[k] + UAV_factor * costMatrix[j][k] <= M * (
                            1 - gp.quicksum(Y[i, j, k] for i in N_out_lst if (i, j, k) in P_lst)))

    for k in N_in_lst:
        for j in C_lst:
            if j != k:
                for i in N_out_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(TUAV[k] - TUAV[j] + UAV_factor * costMatrix[i][j] <= (
                                    E_roundtrip_km / drone_speed_kmh) + M * (1 - Y[i, j, k]))

    m.addConstr(T[0] == 0.0)
    m.addConstr(TUAV[0] == 0.0)

    # 时间窗与惩罚
    for j in C_lst:
        m.addConstr(F[j] >= T[j] - M * (1 - gp.quicksum(X[i, j] for i in N_out_lst if i != j)))
        for i in N_out_lst:
            if i != j:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(F[j] >= TUAV[i] + UAV_factor * costMatrix[i][j] - M * (1 - Y[i, j, k]))
        m.addConstr(L[j] >= F[j] - due_time[j])

    # 完美对齐 simulation.py 的口径：
    # 卡车距离 = 坐标直线距离 * 1.5 (路况系数)
    truck_dist_units = gp.quicksum(
        distMatrix[i][j] * unit_per_km * truck_road_factor * X[i, j]
        for i in N_out_lst for j in N_in_lst if i != j
    )

    # 无人机距离 = 坐标直线距离 (无人机不乘路况系数)
    drone_dist_units = gp.quicksum(
        (distMatrix[i][j] * unit_per_km + distMatrix[j][k] * unit_per_km) * Y[i, j, k]
        for i in N_out_lst for j in C_lst for k in N_in_lst if (i, j, k) in P_lst
    )

    late_sum_h = gp.quicksum(L[j] for j in C_lst)

    # 目标函数：卡车 + alpha * 无人机 + lambda * 迟到
    m.setObjective(truck_dist_units + alpha * drone_dist_units + lambda_late * late_sum_h, GRB.MINIMIZE)

    print(">>> 开始调用 Gurobi 求解...")
    m.optimize()

    if m.SolCount <= 0:
        print("未找到可行解！请增加 time_limit 或检查数据。")
        return

    # 解析路线
    truck_route_idx = [0]
    curr = 0
    while curr != c + 1:
        for j in N_in_lst:
            if X[curr, j].X > 0.5:
                truck_route_idx.append(j)
                curr = j
                break

    drone_triplets = []
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst and Y[i, j, k].X > 0.5:
                    drone_triplets.append((i, j, k))

    # --- 完美打印，供直接写进论文表格 ---
    print(f"\n=========================================")
    print(f"       经典伴飞模式 (FSTSP) 静态结果       ")
    print(f"=========================================")
    print(f"目标函数总成本 (Cost): {m.ObjVal:.3f}")
    print(f"卡车行驶距离 (Units): {truck_dist_units.getValue():.3f}")
    print(f"无人机飞行距离 (Units): {drone_dist_units.getValue():.3f}")
    print(f"系统总迟到时间 (Hours): {late_sum_h.getValue():.3f}")
    print(f"=========================================\n")

    # --- 专属的可视化引擎（彻底消灭基站，只画车和飞机） ---
    plt.figure(figsize=(10, 8), dpi=150)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 中心仓库
    plt.scatter(corX[0], corY[0], c='yellow', marker='s', s=150, edgecolors='black', zorder=6, label='中心仓库 (Depot)')

    # 客户点
    for i in C_lst:
        plt.scatter(corX[i], corY[i], c='blue', s=40, zorder=5)
        plt.text(corX[i] + 0.6, corY[i] + 0.6, str(idx2id[i]), fontsize=9, color='dimgray')

    # 卡车路线 (红色实线)
    for i in range(len(truck_route_idx) - 1):
        n1 = truck_route_idx[i]
        n2 = truck_route_idx[i + 1]
        plt.plot([corX[n1], corX[n2]], [corY[n1], corY[n2]], c='red', lw=2.5, alpha=0.9, zorder=3,
                 label='卡车路径' if i == 0 else "")

    # 无人机伴飞路线 (浅蓝色双向箭头/虚线)
    for idx, (i, j, k) in enumerate(drone_triplets):
        # i(起飞) -> j(客户)
        plt.plot([corX[i], corX[j]], [corY[i], corY[j]], c='skyblue', ls='--', lw=2, zorder=4,
                 label='无人机路径' if idx == 0 else "")
        # j(客户) -> k(降落)
        plt.plot([corX[j], corX[k]], [corY[j], corY[k]], c='skyblue', ls='--', lw=2, zorder=4)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=10)
    plt.title(f"经典伴飞模式 (FSTSP) 路由规划图 - 总成本: {m.ObjVal:.1f}", fontsize=14)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 【请在这里替换成你的 15 节点或 25 节点的 _promise.csv 文件路径！】
    test_csv = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\nodes_25_seed2023_20260129_164341_promise.csv"  # ⚠️ 千万不要放200的，放一个15或25的！

    # 直接运行即可！
    solve_fstsp_static_and_plot(test_csv, time_limit=1800)