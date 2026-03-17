# -*- coding: utf-8 -*-
"""
经典伴飞模式 (FSTSP) 动态重规划评估脚本 (带动态可视化)
- 补齐了 system_time (总完成时间) 和 total_late (总迟到)
- 支持 events.csv 在线扰动
- 为每个动态决策点 (Scene) 自动生成路由规划图
"""

import pandas as pd
import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def load_events(events_path):
    """加载 events.csv 并按时间分组"""
    if not os.path.exists(events_path):
        return {}
    df_ev = pd.read_csv(events_path)
    groups = {}
    for _, row in df_ev.iterrows():
        t = float(row['EVENT_TIME'])
        t_key = round(t, 6)
        if t_key not in groups:
            groups[t_key] = []
        groups[t_key].append({
            'NODE_ID': int(row['NODE_ID']),
            'NEW_X': float(row['NEW_X']),
            'NEW_Y': float(row['NEW_Y'])
        })
    return groups


def plot_fstsp_scene(scene_name, cost, start_pos, depot_pos, nodes_dict, unserved_ids, truck_route_ids, drone_triplets):
    """专属动态可视化引擎：绘制当前决策点的残局路由"""
    plt.figure(figsize=(10, 8), dpi=120)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 终点仓库 (黄方块)
    plt.scatter(depot_pos[0], depot_pos[1], c='yellow', marker='s', s=150, edgecolors='black', zorder=6,
                label='中心仓库 (Depot)')

    # 2. 卡车当前位置 (红星星，仅在动态场景显示)
    if start_pos != depot_pos:
        plt.scatter(start_pos[0], start_pos[1], c='red', marker='*', s=200, edgecolors='black', zorder=7,
                    label='卡车当前位置')

    # 3. 客户点 (只画未服务的)
    for nid in unserved_ids:
        cx, cy = nodes_dict[nid]['X'], nodes_dict[nid]['Y']
        plt.scatter(cx, cy, c='blue', s=40, zorder=5)
        plt.text(cx + 0.6, cy + 0.6, str(nid), fontsize=9, color='dimgray')

    # 获取坐标的辅助函数
    def get_coord(nid):
        if nid == -1: return start_pos  # 虚拟起点
        if nid == 0: return depot_pos  # 终点仓库
        return (nodes_dict[nid]['X'], nodes_dict[nid]['Y'])

    # 4. 卡车路线 (红色实线)
    for i in range(len(truck_route_ids) - 1):
        n1 = truck_route_ids[i]
        n2 = truck_route_ids[i + 1]
        p1, p2 = get_coord(n1), get_coord(n2)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='red', lw=2.5, alpha=0.9, zorder=3,
                 label='卡车路径' if i == 0 else "")

    # 5. 无人机伴飞路线 (浅蓝色双向虚线)
    for idx, (i, j, k) in enumerate(drone_triplets):
        p_i, p_j, p_k = get_coord(i), get_coord(j), get_coord(k)
        # 起飞 -> 客户
        plt.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], c='skyblue', ls='--', lw=2, zorder=4,
                 label='无人机路径' if idx == 0 else "")
        # 客户 -> 降落
        plt.plot([p_j[0], p_k[0]], [p_j[1], p_k[1]], c='skyblue', ls='--', lw=2, zorder=4)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=10)
    plt.title(f"{scene_name} FSTSP 残局路由图 - 后缀Cost: {cost:.1f}", fontsize=14)
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def solve_fstsp_subproblem(
        nodes_dict, unserved_ids, start_pos, start_time, depot_pos,
        E_roundtrip_km=10.0, truck_speed_kmh=30.0, truck_road_factor=1.5,
        drone_speed_kmh=60.0, alpha=0.3, lambda_late=50.0,
        time_limit=1800.0, mip_gap=0.0, unit_per_km=5.0, verbose=0, scene_name="Scene"
):
    c = len(unserved_ids)
    nodeNum = c + 2

    corX = [0.0] * nodeNum
    corY = [0.0] * nodeNum
    due_time = [999999.0] * nodeNum
    idx2id = {}

    # 0 = 虚拟起点 (当前卡车位置)
    idx2id[0] = -1
    corX[0], corY[0] = start_pos[0], start_pos[1]

    # 1 to c = 未服务客户点
    for i, nid in enumerate(unserved_ids):
        idx = i + 1
        idx2id[idx] = nid
        corX[idx] = nodes_dict[nid]['X']
        corY[idx] = nodes_dict[nid]['Y']
        due_time[idx] = nodes_dict[nid]['DUE']

    # c+1 = 中心终点
    idx2id[c + 1] = 0
    corX[c + 1], corY[c + 1] = depot_pos[0], depot_pos[1]

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

    m = gp.Model('FSTSP_Subproblem')
    m.setParam('OutputFlag', verbose)
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

    # 核心路由约束
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

    # 时空同步约束
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

    m.addConstr(T[0] == start_time)
    m.addConstr(TUAV[0] == start_time)

    # 软时间窗惩罚
    for j in C_lst:
        m.addConstr(F[j] >= T[j] - M * (1 - gp.quicksum(X[i, j] for i in N_out_lst if i != j)))
        for i in N_out_lst:
            if i != j:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(F[j] >= TUAV[i] + UAV_factor * costMatrix[i][j] - M * (1 - Y[i, j, k]))
        m.addConstr(L[j] >= F[j] - due_time[j])

    truck_dist_units = gp.quicksum(
        distMatrix[i][j] * unit_per_km * truck_road_factor * X[i, j] for i in N_out_lst for j in N_in_lst if i != j)
    drone_dist_units = gp.quicksum(
        (distMatrix[i][j] * unit_per_km + distMatrix[j][k] * unit_per_km) * Y[i, j, k] for i in N_out_lst for j in C_lst
        for k in N_in_lst if (i, j, k) in P_lst)
    late_sum_h = gp.quicksum(L[j] for j in C_lst)

    m.setObjective(truck_dist_units + alpha * drone_dist_units + lambda_late * late_sum_h, GRB.MINIMIZE)
    m.optimize()

    if m.SolCount <= 0:
        return None

    # 解析路线与真实节点映射
    truck_route_idx = [0]
    curr = 0
    while curr != c + 1:
        for j in N_in_lst:
            if X[curr, j].X > 0.5:
                truck_route_idx.append(j)
                curr = j
                break

    truck_route_ids = [idx2id[n] for n in truck_route_idx]

    drone_triplets = []
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst and Y[i, j, k].X > 0.5:
                    drone_triplets.append((idx2id[i], idx2id[j], idx2id[k]))

    # 提取真实 ID 的排程时刻
    node_arrival = {}
    for idx in range(1, c + 1):
        nid = idx2id[idx]
        node_arrival[nid] = F[idx].X

    res_dict = {
        'cost': m.ObjVal,
        'truck_dist': truck_dist_units.getValue(),
        'drone_dist': drone_dist_units.getValue(),
        'system_time': T[c + 1].X,
        'total_late': late_sum_h.getValue(),
        'node_arrival': node_arrival,
        'truck_route_ids': truck_route_ids,
        'drone_triplets': drone_triplets
    }

    # 调用画图
    plot_fstsp_scene(scene_name, res_dict['cost'], start_pos, depot_pos, nodes_dict, unserved_ids, truck_route_ids,
                     drone_triplets)

    return res_dict


def run_dynamic_fstsp(csv_path, events_path, decision_times=[1.0, 2.0], time_limit=1800.0):
    """动态滚动主框架"""
    print(f"\n================= 经典伴飞 (FSTSP) 动态环境 ==================")
    df = pd.read_csv(csv_path)
    df_cust = df[df['NODE_TYPE'].str.lower() == 'customer'].reset_index(drop=True)
    depot_row = df[df['NODE_TYPE'].str.lower() == 'central'].iloc[0]

    depot_pos = (float(depot_row['ORIG_X']), float(depot_row['ORIG_Y']))

    nodes_dict = {}
    for i in range(len(df_cust)):
        nid = int(df_cust.loc[i, 'NODE_ID'])
        nodes_dict[nid] = {
            'X': float(df_cust.loc[i, 'ORIG_X']),
            'Y': float(df_cust.loc[i, 'ORIG_Y']),
            'DUE': float(df_cust.loc[i, 'EFFECTIVE_DUE']) if 'EFFECTIVE_DUE' in df_cust.columns else float(
                df_cust.loc[i, 'DUE_TIME'])
        }

    events_grouped = load_events(events_path)
    unserved_ids = list(nodes_dict.keys())

    current_truck_pos = depot_pos

    # [场景 0: 静态规划]
    print(f"\n>>> Scene 0: t=0.0h 静态初始求解...")
    res = solve_fstsp_subproblem(
        nodes_dict, unserved_ids, current_truck_pos, 0.0, depot_pos,
        time_limit=time_limit, verbose=0, scene_name="Scene 0"
    )

    if res is None:
        print("Scene 0 无解！")
        return
    print(
        f"    [Scene 0 结果] Cost: {res['cost']:.3f} | System Time: {res['system_time']:.3f}h | Total Late: {res['total_late']:.3f}h")

    planned_arrival = res['node_arrival']

    # 动态滚动
    for scene_idx, t_dec in enumerate(decision_times, start=1):
        print(f"\n>>> Scene {scene_idx}: 决策时刻 t={t_dec:.2f}h")

        # 1. 甄别已服务客户
        served_this_round = []
        for nid in list(unserved_ids):
            if planned_arrival.get(nid, 999) <= t_dec:
                served_this_round.append(nid)
                unserved_ids.remove(nid)

        # 2. 粗略插值更新卡车当前物理位置
        if served_this_round:
            last_served = served_this_round[-1]
            current_truck_pos = (nodes_dict[last_served]['X'], nodes_dict[last_served]['Y'])

        # 3. 注入位置变更事件
        evs = events_grouped.get(round(t_dec, 6), [])
        req_count = 0
        for ev in evs:
            nid = ev['NODE_ID']
            if nid in unserved_ids:
                nodes_dict[nid]['X'] = ev['NEW_X']
                nodes_dict[nid]['Y'] = ev['NEW_Y']
                req_count += 1

        if req_count == 0:
            print(f"    [跳过] t={t_dec}h 时无有效扰动，跳过重规划。")
            continue

        print(f"    已接受 {req_count} 个客户的在线位置变更。开始 Gurobi 重规划残局...")

        # 4. 残局重规划
        res_sub = solve_fstsp_subproblem(
            nodes_dict, unserved_ids, current_truck_pos, t_dec, depot_pos,
            time_limit=time_limit, verbose=0, scene_name=f"Scene {scene_idx}"
        )

        if res_sub is None:
            print("    [警告] 残局无解，FSTSP 陷入时空瘫痪！")
            break

        print(
            f"    [残局结果] Suffix Cost: {res_sub['cost']:.3f} | End System Time: {res_sub['system_time']:.3f}h | Suffix Late: {res_sub['total_late']:.3f}h")
        planned_arrival.update(res_sub['node_arrival'])

    print(f"\n================= 动态测试结束 ==================")


if __name__ == "__main__":
    # 【注意】：FSTSP 变量呈立方级爆炸，千万不要用 50/100/200 的数据跑动态！
    # 请务必替换为你生成的 N=25 的数据路径
    test_nodes = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\nodes_25_seed2023_20260129_164341_promise.csv"
    test_events = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\events_25_seed2023_20260129_164341.csv"

    run_dynamic_fstsp(test_nodes, test_events, decision_times=[1.0, 2.0], time_limit=1800.0)
