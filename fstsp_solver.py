# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def solve_fstsp_return_from_df(
        df: pd.DataFrame,
        E_roundtrip_km: float = 10.0,
        truck_speed_kmh: float = 30.0,
        truck_road_factor: float = 1.5,
        drone_speed_kmh: float = 60.0,
        alpha: float = 0.3,
        lambda_late: float = 50.0,
        time_limit: float = 1800.0,
        mip_gap: float = 0.0,
        unit_per_km: float = 5.0,
        start_node: int = None,
        start_time_h: float = 0.0,
        verbose: int = 0
):
    """
    纯粹的 FSTSP (伴飞模式) 求解器。
    输入：DataFrame (包含 truck_pos, central, customer)
    输出：字典格式，包含 route 和 drone_assign，完全兼容你的 ALNS 框架动态重规划需求。
    """
    t_start = time.time()

    # 解析 df，提取关键节点
    start_row = df[df['NODE_TYPE'] == 'truck_pos'].iloc[0]
    depot_row = df[df['NODE_TYPE'] == 'central'].iloc[0]
    df_cust = df[df['NODE_TYPE'] == 'customer']

    c = len(df_cust)
    nodeNum = c + 2

    corX = [0.0] * nodeNum
    corY = [0.0] * nodeNum
    due_time = [999999.0] * nodeNum
    idx2id = {}

    # 0 索引：虚拟起点
    idx2id[0] = int(start_row['NODE_ID'])
    corX[0], corY[0] = float(start_row['ORIG_X']), float(start_row['ORIG_Y'])

    # 1 到 c：客户点
    for i, (_, row) in enumerate(df_cust.iterrows()):
        idx = i + 1
        idx2id[idx] = int(row['NODE_ID'])
        corX[idx] = float(row['ORIG_X'])
        corY[idx] = float(row['ORIG_Y'])
        due_time[idx] = float(row['EFFECTIVE_DUE'])

    # c + 1：终点 (Depot)
    idx2id[c + 1] = int(depot_row['NODE_ID'])
    corX[c + 1], corY[c + 1] = float(depot_row['ORIG_X']), float(depot_row['ORIG_Y'])

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

    m = gp.Model('FSTSP_MILP')
    m.setParam('OutputFlag', verbose)
    m.setParam('TimeLimit', float(time_limit))
    m.setParam('MIPGap', float(mip_gap))

    M = 100000.0
    X = m.addVars(nodeNum, nodeNum, vtype=GRB.BINARY, name="X")
    Y = m.addVars(nodeNum, nodeNum, nodeNum, vtype=GRB.BINARY, name="Y")
    U = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=1.0, ub=c + 2, name="U")
    T = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    TUAV = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="TUAV")
    F = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    L = m.addVars(nodeNum, vtype=GRB.CONTINUOUS, lb=0.0, name="Late")

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
                        m.addConstr(gp.quicksum(X[h, i] for h in N_out_lst if h != i) +
                                    gp.quicksum(X[l, k] for l in C_lst if l != k) >= 2 * Y[i, j, k])

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

    m.addConstr(T[0] == start_time_h)
    m.addConstr(TUAV[0] == start_time_h)

    for j in C_lst:
        m.addConstr(F[j] >= T[j] - M * (1 - gp.quicksum(X[i, j] for i in N_out_lst if i != j)))
        for i in N_out_lst:
            if i != j:
                for k in N_in_lst:
                    if (i, j, k) in P_lst:
                        m.addConstr(F[j] >= TUAV[i] + UAV_factor * costMatrix[i][j] - M * (1 - Y[i, j, k]))
        m.addConstr(L[j] >= F[j] - due_time[j])

    # 为了和你的 ALNS 框架完全对齐，我们在目标函数里也使用 unit (而非 km) 为口径
    truck_dist_units = gp.quicksum(
        distMatrix[i][j] * unit_per_km * truck_road_factor * X[i, j] for i in N_out_lst for j in N_in_lst if i != j)
    drone_dist_units = gp.quicksum(
        (distMatrix[i][j] * unit_per_km + distMatrix[j][k] * unit_per_km) * Y[i, j, k] for i in N_out_lst for j in C_lst
        for k in N_in_lst if (i, j, k) in P_lst)
    late_sum_h = gp.quicksum(L[j] for j in C_lst)

    m.setObjective(truck_dist_units + alpha * drone_dist_units + lambda_late * late_sum_h, GRB.MINIMIZE)
    m.optimize()

    runtime = time.time() - t_start

    if m.SolCount <= 0:
        return {"sol_count": 0, "runtime_sec": runtime}

    # 提取卡车路线
    truck_route_idx = [0]
    curr = 0
    while curr != c + 1:
        for j in N_in_lst:
            if X[curr, j].X > 0.5:
                truck_route_idx.append(j)
                curr = j
                break

    truck_route_ids = [idx2id[n] for n in truck_route_idx]

    # 将无人机起降的 (i, j, k) 转换为 ALNS 能读懂的 drone_assign: {base_node: [customer_nodes]}
    # 在 FSTSP 中，卡车就是基站，起飞点 i 即为 base。
    fstsp_triplets = []
    drone_assign = {}  # 保留兼容性
    for i in N_out_lst:
        for j in C_lst:
            for k in N_in_lst:
                if (i, j, k) in P_lst and Y[i, j, k].X > 0.5:
                    launch_node_id = idx2id[i]
                    cust_node_id = idx2id[j]
                    rendezvous_id = idx2id[k]

                    fstsp_triplets.append((launch_node_id, cust_node_id, rendezvous_id))
                    drone_assign.setdefault(launch_node_id, []).append(cust_node_id)

    return {
        "status": m.Status,
        "sol_count": m.SolCount,
        "route": truck_route_ids,
        "drone_assign": drone_assign,
        "fstsp_triplets": fstsp_triplets,  # <--- 新增这行输出
        "obj": m.ObjVal,
        "runtime_sec": runtime
    }