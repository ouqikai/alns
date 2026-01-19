# -*- coding: utf-8 -*-
"""
静态卡车-无人机（基站往返）协同配送 MILP（Gurobi）
数据字段（CSV）：
  NODE_ID, NODE_TYPE, ORIG_X, ORIG_Y, DEMAND, READY_TIME, DUE_TIME
其中：
  NODE_TYPE ∈ {central, base, customer}

关键：距离单位换算
- 你的数据集坐标是“单位坐标”，例如 5单位 = 1km
- 本脚本通过 --unit_per_km 参数做换算：dist_km = dist_unit / unit_per_km
- 卡车道路折线系数：dist_truck_km = dist_km * truck_road_factor（默认1.5，用于近似道路绕行，影响卡车距离与行驶时间）
- 速度 truck_speed / drone_speed 以 km/h 输入
- E 为无人机最大“往返距离”（km），约束：2*dist_km(base,customer) <= E

模型说明（静态验证版）：
- 每个客户 c：要么卡车直送（z_c=1），要么由某个基站 b 的无人机服务（y_{b,c}=1）
- 无人机服务形式：b -> c -> b（仅用 E 约束往返距离）
- 若 y_{b,c}=1，则卡车必须访问该基站 b（u_b=1），且客户完成时间 >= 卡车到达基站时间 + 无人机去程时间
- 软时间窗：迟到 L_c >= F_c - DUE_TIME_c
- 目标：min truck_dist + alpha * drone_dist + lambda * sum(L_c)

注意：未建无人机多架并行/排队调度（后续可扩展）。
"""

import argparse
import math
from typing import Dict, Tuple, List, Optional, Set, Any, Iterable

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
from datetime import datetime

# 可选：可视化依赖（没有 matplotlib 也能跑 MILP）
try:
    import matplotlib
    # 说明：不强制切换 backend，优先兼容本地 PyCharm 弹窗；无界面环境下也可仅保存图片。
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    Line2D = None

def euclid(x1, y1, x2, y2) -> float:
    return math.hypot(x1 - x2, y1 - y2)




def plot_solution(nodes: Dict[int, Dict],
                  depot: int,
                  bases: List[int],
                  customers: List[int],
                  truck_route: List[int],
                  drone_assign: Dict[int, List[int]],
                  out_path: str,
                  title: str = "Static MILP Truck-Drone Solution",
                  show: bool = False):

    if plt is None:
        print("[PLOT] 未安装 matplotlib，跳过可视化。")
        return

    def _setup_cn_font():
        # 按优先级依次尝试：微软雅黑 / 黑体 / Arial Unicode（不同机器不一样）
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK SC'
        ]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

    # 在 plot_solution() 一开始调用一次
    _setup_cn_font()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

    # --- 坐标范围（自动 padding + 对齐到 10）---
    xs = [float(nodes[i]["x"]) for i in nodes.keys()]
    ys = [float(nodes[i]["y"]) for i in nodes.keys()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = max(5.0, 0.05 * span)

    def _round10_floor(v): return math.floor(v / 10.0) * 10.0
    def _round10_ceil(v): return math.ceil(v / 10.0) * 10.0

    x0 = _round10_floor(min_x - pad)
    x1 = _round10_ceil(max_x + pad)
    y0 = _round10_floor(min_y - pad)
    y1 = _round10_ceil(max_y + pad)

    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", linewidth=0.6, color="gray", alpha=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    # --- 服务模式集合 ---
    drone_customers = set()
    for b, lst in drone_assign.items():
        for c in lst:
            drone_customers.add(c)
    truck_customers = set(customers) - drone_customers

    # --- 画 central & bases ---
    cx, cy = nodes[depot]["x"], nodes[depot]["y"]
    ax.scatter([cx], [cy], marker="s", s=120, c="yellow", edgecolors="black", zorder=6)
    for b in bases:
        bx, by = nodes[b]["x"], nodes[b]["y"]
        ax.scatter([bx], [by], marker="*", s=180, c="yellow", edgecolors="black", zorder=6)

    # --- 画 customers ---
    if truck_customers:
        tx = [nodes[c]["x"] for c in truck_customers]
        ty = [nodes[c]["y"] for c in truck_customers]
        ax.scatter(tx, ty, facecolors="none", edgecolors="blue", s=35, linewidths=1.6, zorder=5)

    if drone_customers:
        dx = [nodes[c]["x"] for c in drone_customers]
        dy = [nodes[c]["y"] for c in drone_customers]
        ax.scatter(dx, dy, c="blue", s=30, zorder=5)

    # --- 画卡车路径（红线 + 箭头）---
    # 仅画 route 中相邻边
    for i in range(len(truck_route) - 1):
        a = truck_route[i]
        b = truck_route[i + 1]
        x1, y1 = nodes[a]["x"], nodes[a]["y"]
        x2, y2 = nodes[b]["x"], nodes[b]["y"]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.8, alpha=0.9, zorder=3)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2, alpha=0.9))

    # --- 画无人机分配（浅蓝）---
    for b, lst in drone_assign.items():
        bx, by = nodes[b]["x"], nodes[b]["y"]
        for c in lst:
            cx, cy = nodes[c]["x"], nodes[c]["y"]
            ax.plot([bx, cx], [by, cy], color="skyblue", linewidth=1.2, alpha=0.9, zorder=3)
            ax.annotate("", xy=(cx, cy), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="-|>", color="skyblue", lw=1.0, alpha=0.9))

    # --- 图例 ---
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=9, label="中心仓库"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=11, label="基站"),
        Line2D([0], [0], marker="o", color="blue", markerfacecolor="white",
               markeredgecolor="blue", markersize=7, label="卡车客户"),
        Line2D([0], [0], marker="o", color="blue", markerfacecolor="blue",
               markeredgecolor="blue", markersize=7, label="无人机客户"),
        Line2D([0], [0], color="red", lw=1.8, label="卡车路径"),
        Line2D([0], [0], color="skyblue", lw=1.2, label="无人机路径"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=160)
        print(f"[PLOT] saved: {out_path}")

    if show:
        # 如果你希望在 PyCharm 里弹窗显示，把 --plot_show 设为 1
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)

def solve_milp_return_from_df(
        df: pd.DataFrame,
        *,
        E_roundtrip_km: float,
        truck_speed_kmh: float,
        truck_road_factor: float,
        drone_speed_kmh: float,
        alpha: float,
        lambda_late: float,
        time_limit: float,
        mip_gap: float,
        unit_per_km: float,
        allowed_bases: Optional[Set[int]] = None,
        must_visit_bases: Optional[Set[int]] = None,
        drone_only_bases: Optional[Set[int]] = None,
        drone_only_onhand: Optional[Dict[int, Iterable[int]]] = None,
        force_truck_customers: Optional[Set[int]] = None,
        allow_depot_as_base: bool = True,
        verbose: int = 0,
        save_iis_path: str = "model_iis.ilp",
        start_node: Optional[int] = None,      # 动态起点 NODE_ID（允许 -1 这种虚拟点）
        end_node: Optional[int] = None,        # 动态终点，默认 central
        start_time_h: float = 0.0,             # 决策时刻 t_dec（绝对时间）
) -> Dict[str, Any]:
    """
    将原 build_and_solve 的“建模+求解+解析”封装成可复用函数（返回结果，不做打印/画图）。
    - allowed_bases: 只允许这些基站作为无人机起点（用于“禁止已错过基站”）；None 表示不限制
    - force_truck_customers: 强制这些客户必须卡车服务（用于动态系统的兜底/硬约束）
    - allow_depot_as_base: 是否允许 central 也派无人机（静态脚本原本允许）
    返回：
      {
        "status": int, "status_name": str, "sol_count": int,
        "route": [NODE_ID...],
        "drone_assign": {base_node_id: [cust_node_id...]},
        "obj": float|None,
        "truck_km": float|None, "drone_km": float|None, "late_sum": float|None,
        "base_cost_no_late": float|None,
        "runtime_sec": float,
        "mip_gap": float|None
      }
    """
    if truck_road_factor <= 0:
        raise ValueError("truck_road_factor 必须 > 0")
    if unit_per_km <= 0:
        raise ValueError("unit_per_km 必须 > 0")

    required_cols = {"NODE_ID", "NODE_TYPE", "ORIG_X", "ORIG_Y", "READY_TIME", "DUE_TIME"}
    miss = required_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"DataFrame 缺少字段: {miss}, 实际字段={list(df.columns)}")

    # ---------- 解析 nodes ----------
    nodes: Dict[int, Dict] = {}
    for _, row in df.iterrows():
        nid = int(row["NODE_ID"])
        due_prom = float(row["DUE_TIME"])  # 承诺截止：永远用 DUE_TIME
        due_eff = due_prom  # 有效截止：若有 EFFECTIVE_DUE 则用它

        if "EFFECTIVE_DUE" in df.columns:
            v = row.get("EFFECTIVE_DUE", None)
            try:
                if v is not None and (not (isinstance(v, float) and math.isnan(v))):
                    due_eff = float(v)
            except Exception:
                pass

        nodes[nid] = {
            "type": str(row["NODE_TYPE"]).strip(),
            "x": float(row["ORIG_X"]),
            "y": float(row["ORIG_Y"]),
            "ready": float(row["READY_TIME"]),
            "due": float(due_eff),  # 继续保持旧行为：L 用 effective due
            "due_prom": float(due_prom),  # 新增：prom due
            "demand": float(row["DEMAND"]) if "DEMAND" in df.columns else 0.0
        }

    depot_list = [nid for nid, nd in nodes.items() if nd["type"] == "central"]
    if len(depot_list) != 1:
        raise RuntimeError(f"central 节点应当且仅当 1 个，实际={depot_list}")
    depot = depot_list[0]

    bases_all = sorted([nid for nid, nd in nodes.items() if nd["type"] == "base"])
    customers = sorted([nid for nid, nd in nodes.items() if nd["type"] == "customer"])
    if start_node is None:
        start_node = depot
    if end_node is None:
        end_node = depot
    start_node = int(start_node)
    end_node = int(end_node)

    # ---------- allowed_bases：限制可用基站 ----------
    if allowed_bases is not None:
        allowed_bases = set(int(x) for x in allowed_bases)
        bases = [b for b in bases_all if b in allowed_bases]
    else:
        bases = bases_all

    # 无人机起点集合：是否允许 depot
    drone_bases: List[int] = []
    if allow_depot_as_base:
        # depot 是否需要被 allowed_bases 限制：一般不需要；若你希望限制，则在外层传 allowed_bases 并自行包含 depot
        drone_bases.append(depot)
    drone_bases += bases

    V = [depot] + bases + customers
    if start_node != depot and start_node not in V:
        V = [start_node] + V
    nV = len(V)

    # ---------- 预计算卡车距离/时间（km/h 口径） ----------
    dT_km: Dict[Tuple[int, int], float] = {}
    tT_h: Dict[Tuple[int, int], float] = {}
    max_dT_km = 0.0
    for i in V:
        xi, yi = nodes[i]["x"], nodes[i]["y"]
        for j in V:
            if i == j:
                continue
            xj, yj = nodes[j]["x"], nodes[j]["y"]
            dij_unit = euclid(xi, yi, xj, yj)
            dij_km = (dij_unit / unit_per_km) * truck_road_factor
            dT_km[(i, j)] = dij_km
            tT_h[(i, j)] = dij_km / truck_speed_kmh
            max_dT_km = max(max_dT_km, dij_km)
    drone_only_bases = set(int(x) for x in (drone_only_bases or []))
    _doo = drone_only_onhand or {}
    drone_only_onhand = {int(b): set(int(c) for c in cs) for b, cs in _doo.items()}
    # ---------- 无人机可行对 ----------
    feas_pairs: List[Tuple[int, int]] = []
    dD_km: Dict[Tuple[int, int], float] = {}
    tD_h: Dict[Tuple[int, int], float] = {}
    for b in drone_bases:
        xb, yb = nodes[b]["x"], nodes[b]["y"]
        cand_customers = customers
        if b in drone_only_bases:
            cand_customers = sorted([c for c in customers if c in drone_only_onhand.get(b, set())])
        for c in cand_customers:
            xc, yc = nodes[c]["x"], nodes[c]["y"]
            dbc_unit = euclid(xb, yb, xc, yc)
            dbc_km = dbc_unit / unit_per_km
            rt_km = 2.0 * dbc_km
            if rt_km <= E_roundtrip_km + 1e-9:
                feas_pairs.append((b, c))
                dD_km[(b, c)] = dbc_km
                tD_h[(b, c)] = dbc_km / drone_speed_kmh

    max_due = max(nodes[c]["due"] for c in customers) if customers else 0.0
    M_time = (nV * max_dT_km / truck_speed_kmh) + max_due + 10.0

    # ---------- 建模 ----------
    m = gp.Model("static_truck_drone_milp")
    m.Params.OutputFlag = 1 if int(verbose) else 0
    m.Params.TimeLimit = float(time_limit)
    m.Params.MIPGap = float(mip_gap)

    x = m.addVars([(i, j) for i in V for j in V if i != j], vtype=GRB.BINARY, name="x")
    u = m.addVars([i for i in V if i != depot], vtype=GRB.BINARY, name="u")
    z = m.addVars(customers, vtype=GRB.BINARY, name="z")
    y = m.addVars(feas_pairs, vtype=GRB.BINARY, name="y")
    if int(verbose) >= 2:
        print("[MILP-DBG] feas_pairs_for_10_11=", [(b, c) for (b, c) in feas_pairs if c in (10, 11)])

    T = m.addVars(V, vtype=GRB.CONTINUOUS, lb=0.0, name="T")
    # ---------- 起点时间 ----------
    if start_node == depot:
        m.addConstr(T[depot] == 0.0, name="T_depot_0")
    else:
        m.addConstr(T[start_node] == float(start_time_h), name="T_start_fix")
    # ---------- visited drone-only base：固定时间为决策时刻，避免“时间倒流” ----------
    # drone_only_bases：你前面新增传进来的集合（visited 且 drone-only 的基站）
    if drone_only_bases:
        for b in drone_only_bases:
            b = int(b)
            # 只对模型里存在的节点生效，并避免和 start_node 重复固定
            if (b in V) and (b != start_node):
                m.addConstr(T[b] == float(start_time_h), name=f"T_fix_drone_only_{b}")

    F = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0.0, name="F")
    L = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0.0, name="L")
    Lp = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=0.0, name="Lp")  # 承诺窗迟到
    p = m.addVars([i for i in V if i != depot], vtype=GRB.CONTINUOUS, lb=0.0, ub=nV - 1, name="p")
    if must_visit_bases:
        mv = set(int(x) for x in must_visit_bases)
        # 仅强制“当前确实可能用于无人机派送”的基站：否则会出现‘基站没有客户也被硬访问’
        feas_base_set = set(int(b) for (b, _) in feas_pairs)
        for b in mv:
            if b in V and b != depot and b != start_node:
                if b not in feas_base_set:
                    continue
                m.addConstr(u[b] == 1, name=f"must_visit_base_{b}")

    # ---------- 约束：起终点出入度（静态闭环 / 动态开路径）----------
    if start_node == depot:
        # 静态：闭环 tour（和你原来一样）
        m.addConstr(gp.quicksum(x[depot, j] for j in V if j != depot) == 1, name="depot_out")
        m.addConstr(gp.quicksum(x[i, depot] for i in V if i != depot) == 1, name="depot_in")
    else:
        # 动态：从 start_node 出发，最后到 depot（开路径）
        m.addConstr(gp.quicksum(x[start_node, j] for j in V if j != start_node) == 1, name="start_out")
        m.addConstr(gp.quicksum(x[i, start_node] for i in V if i != start_node) == 0, name="start_in")

        m.addConstr(gp.quicksum(x[i, depot] for i in V if i != depot) == 1, name="end_in")
        m.addConstr(gp.quicksum(x[depot, j] for j in V if j != depot) == 0, name="end_out")

    # ---------- 约束：流平衡 ----------
    for i in V:
        if i == depot:
            continue
        out_i = gp.quicksum(x[i, j] for j in V if j != i)
        in_i = gp.quicksum(x[j, i] for j in V if j != i)

        if start_node != depot and i == start_node:
            # start：出=1，入=0（并强制它被“访问”）
            m.addConstr(out_i == 1, name=f"flow_out_start_{i}")
            m.addConstr(in_i == 0, name=f"flow_in_start_{i}")
            m.addConstr(u[i] == 1, name=f"visit_start_{i}")
        else:
            m.addConstr(out_i == u[i], name=f"flow_out_{i}")
            m.addConstr(in_i == u[i], name=f"flow_in_{i}")

    # ---------- 客户访问：u[c] == z[c] ----------
    for c in customers:
        m.addConstr(u[c] == z[c], name=f"visit_customer_equals_truck_{c}")
    for c in customers:
        m.addConstr(Lp[c] >= F[c] - nodes[c]["due_prom"], name=f"late_prom_{c}")

    # ---------- 每个客户必须被服务 ----------
    y_by_c: Dict[int, List[Tuple[int, int]]] = {c: [] for c in customers}
    for (b, c) in feas_pairs:
        y_by_c[c].append((b, c))
    for c in customers:
        m.addConstr(z[c] + gp.quicksum(y[pair] for pair in y_by_c[c]) == 1, name=f"serve_once_{c}")

    # ---------- 若 y[b,c]=1，则基站 b 必须被访问（b!=depot） ----------
    for (b, c) in feas_pairs:
        if b != depot and b not in drone_only_bases:
            m.addConstr(y[b, c] <= u[b], name=f"base_visit_if_drone_{b}_{c}")

    # ---------- 强制卡车客户（新增） ----------
    if force_truck_customers:
        ft = set(int(x) for x in force_truck_customers)
        for c in customers:
            if c in ft:
                m.addConstr(z[c] == 1, name=f"force_truck_{c}")

    # ---------- 时间传播 ----------
    for (i, j) in x.keys():
        if j == start_node:
            continue
        m.addConstr(T[j] >= T[i] + tT_h[(i, j)] - M_time * (1 - x[i, j]), name=f"timeprop_{i}_{j}")

    # ---------- 完成时间 ----------
    for c in customers:
        m.addConstr(F[c] >= T[c] - M_time * (1 - z[c]), name=f"F_truck_lb_{c}")
        m.addConstr(F[c] <= T[c] + M_time * (1 - z[c]), name=f"F_truck_ub_{c}")
    for (b, c) in feas_pairs:
        m.addConstr(F[c] >= T[b] + tD_h[(b, c)] - M_time * (1 - y[b, c]), name=f"F_drone_lb_{b}_{c}")
    for c in customers:
        m.addConstr(F[c] >= nodes[c]["ready"], name=f"ready_{c}")

    # ---------- 迟到 ----------
    for c in customers:
        m.addConstr(L[c] >= F[c] - nodes[c]["due"], name=f"late_{c}")

    # ---------- MTZ ----------
    for i in V:
        if i == depot:
            continue
        m.addConstr(p[i] <= (nV - 1) * u[i], name=f"p_ub_visit_{i}")

    for i in V:
        if i == depot:
            continue
        for j in V:
            if j == depot or j == i:
                continue
            m.addConstr(
                p[i] - p[j] + nV * x[i, j] <= (nV - 1) + nV * (1 - u[i]) + nV * (1 - u[j]),
                name=f"mtz_{i}_{j}"
            )

    # ---------- 目标 ----------
    truck_dist = gp.quicksum(dT_km[(i, j)] * x[i, j] for (i, j) in x.keys())
    drone_dist = gp.quicksum(2.0 * dD_km[(b, c)] * y[b, c] for (b, c) in feas_pairs)
    late_eff_pen = gp.quicksum(L[c] for c in customers)
    late_prom_pen = gp.quicksum(Lp[c] for c in customers)

    lambda_late_km = float(lambda_late) / float(unit_per_km)
    lambda_prom = float(kwargs.get("lambda_prom", 0.0)) if isinstance(locals().get("kwargs", None), dict) else 0.0
    lambda_prom_km = float(lambda_prom) / float(unit_per_km)

    m.setObjective(
        truck_dist + alpha * drone_dist
        + lambda_late_km * late_eff_pen
        + lambda_prom_km * late_prom_pen,
        GRB.MINIMIZE
    )

    # ---------- 求解 ----------
    m.optimize()

    status = int(m.Status)
    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
    }.get(status, str(status))

    # 无解/无可行解：直接返回（必要时写 IIS）
    sol_count = int(getattr(m, "SolCount", 0))
    if status == GRB.INFEASIBLE:
        try:
            m.computeIIS()
            m.write(save_iis_path)
        except Exception:
            pass

    if sol_count <= 0:
        return {
            "status": status,
            "status_name": status_name,
            "sol_count": sol_count,
            "route": [],
            "drone_assign": {},
            "obj": None,
            "truck_km": None,
            "drone_km": None,
            "late_sum": None,
            "base_cost_no_late": None,
            "runtime_sec": float(getattr(m, "Runtime", 0.0)),
            "mip_gap": None
        }

    # ---------- 解析 ----------
    obj = float(m.ObjVal)
    truck_val = float(truck_dist.getValue())
    drone_val = float(drone_dist.getValue())
    late_val = float(late_pen.getValue())
    base_cost_no_late = truck_val + alpha * drone_val

    # route：succ 链
    # 从 x 解出后继
    succ = {}
    for (i, j) in x.keys():
        if x[i, j].X > 0.5:
            succ[i] = j

    # 解析路线：静态闭环从 depot；动态开路径从 start_node，终点到 depot
    if start_node == depot:
        route = [depot]
        cur = depot
        for _ in range(len(V) + 5):
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
            if cur == depot:
                break
    else:
        route = [start_node]
        cur = start_node
        for _ in range(len(V) + 5):
            nxt = succ.get(cur, None)
            if nxt is None:
                break
            route.append(nxt)
            cur = nxt
            if cur == depot:
                break
        # 保险：如果没走到 depot，强制补上
        if not route or route[-1] != depot:
            route.append(depot)

    # drone_assign：base -> customers
    drone_assign: Dict[int, List[int]] = {}
    for (b, c) in feas_pairs:
        if y[b, c].X > 0.5:
            drone_assign.setdefault(b, []).append(c)

    # gap（time limit 下可能有）
    gap_val = None
    try:
        gap_val = float(m.MIPGap)
    except Exception:
        pass

    return {
        "status": status,
        "status_name": status_name,
        "sol_count": sol_count,
        "route": route,
        "drone_assign": drone_assign,
        "obj": obj,
        "truck_km": truck_val,
        "drone_km": drone_val,
        "late_sum": late_val,
        "base_cost_no_late": base_cost_no_late,
        "runtime_sec": float(getattr(m, "Runtime", 0.0)),
        "mip_gap": gap_val
    }

def build_and_solve(csv_path: str,
                    E_roundtrip_km: float,
                    truck_speed_kmh: float,
                    truck_road_factor: float,
                    drone_speed_kmh: float,
                    alpha: float,
                    lambda_late: float,
                    time_limit: float,
                    mip_gap: float,
                    unit_per_km: float,
                    verbose: int,
                    plot: int,
                    plot_path: str,
                    plot_show: int):

    # 1) 读数据
    df = pd.read_csv(csv_path)

    # 2) 调 MILP
    res = solve_milp_return_from_df(
        df,
        E_roundtrip_km=E_roundtrip_km,
        truck_speed_kmh=truck_speed_kmh,
        truck_road_factor=truck_road_factor,
        drone_speed_kmh=drone_speed_kmh,
        alpha=alpha,
        lambda_late=lambda_late,
        time_limit=time_limit,
        mip_gap=mip_gap,
        unit_per_km=unit_per_km,
        allowed_bases=None,                 # 静态脚本默认不限制
        must_visit_bases=None,
        force_truck_customers=None,         # 静态脚本默认不强制
        allow_depot_as_base=True,
        verbose=verbose,
    )

    # 3) 无解直接返回
    if res["sol_count"] <= 0:
        print("[WARN] 无可行解/无 incumbent，status=", res["status_name"])
        return

    # 4) 打印结果
    obj = res["obj"]
    truck_val = res["truck_km"]
    drone_val = res["drone_km"]
    late_val = res["late_sum"]
    base_cost_no_late = res["base_cost_no_late"]

    print("\n===== 求解结果（静态 MILP）=====")
    print(f"Status = {res['status_name']}  SolCount={res['sol_count']}  Runtime={res['runtime_sec']:.3f}s  Gap={res.get('mip_gap')}")
    print(f"Obj = {obj:.4f}")
    print(f"truck_dist(km) = {truck_val:.4f}")
    print(f"drone_dist(km) = {drone_val:.4f}  (目标里乘 alpha={alpha})")
    print(f"late_sum(h)    = {late_val:.4f}  (目标里乘 lambda={lambda_late})")
    print(f"base_cost(no_late) = {base_cost_no_late:.4f}")

    route = res["route"]
    drone_assign = res["drone_assign"]
    print("\n[TRUCK] 路线 NODE_ID：", route)
    print("[DRONE] 分配：", drone_assign)

    # 5) 可视化：需要把 nodes/bases/customers/depot 解析出来
    if int(plot) == 1:
        out_png = plot_path.strip() if isinstance(plot_path, str) else ""
        if not out_png:
            out_png = "milp_solution.png"

        # 解析 nodes / depot / bases / customers
        nodes: Dict[int, Dict] = {}
        for _, row in df.iterrows():
            nid = int(row["NODE_ID"])
            due_val = row["DUE_TIME"]
            if "EFFECTIVE_DUE" in df.columns:
                v = row.get("EFFECTIVE_DUE", None)
                try:
                    if v is not None and (not (isinstance(v, float) and math.isnan(v))):
                        due_val = v
                except Exception:
                    pass
            nodes[nid] = {
                "type": str(row["NODE_TYPE"]).strip(),
                "x": float(row["ORIG_X"]),
                "y": float(row["ORIG_Y"]),
                "ready": float(row["READY_TIME"]),
                "due": float(due_val),
                "demand": float(row["DEMAND"]) if "DEMAND" in df.columns else 0.0
            }

        depot_list = [nid for nid, nd in nodes.items() if nd["type"] == "central"]
        if len(depot_list) != 1:
            print("[PLOT][WARN] central 节点应当且仅当 1 个，实际=", depot_list)
            return
        depot = depot_list[0]
        bases = sorted([nid for nid, nd in nodes.items() if nd["type"] == "base"])
        customers = sorted([nid for nid, nd in nodes.items() if nd["type"] == "customer"])

        plot_solution(
            nodes=nodes,
            depot=depot,
            bases=bases,
            customers=customers,
            truck_route=route,
            drone_assign=drone_assign,
            out_path=out_png,
            title="Static MILP Truck-Drone Solution",
            show=bool(plot_show),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument("--E", type=float, required=True, help="无人机最大往返距离 E（km），约束 2*dist_km(base,c) <= E")
    parser.add_argument("--truck_road_factor", type=float, default=1.5,
                        help="卡车道路折线系数：将欧氏距离换算后的 km 再乘以该系数（默认1.5），用于近似道路绕行。")
    parser.add_argument("--truck_speed", type=float, default=30.0, help="卡车速度（km/h）")
    parser.add_argument("--drone_speed", type=float, default=60.0, help="无人机速度（km/h）")
    parser.add_argument("--alpha", type=float, default=0.3, help="无人机距离折算系数 alpha")
    parser.add_argument("--lambda_late", type=float, default=50.0, help="迟到惩罚系数 lambda")
    parser.add_argument("--time_limit", type=float, default=120.0, help="Gurobi 求解时间限制（秒）")
    parser.add_argument("--mip_gap", type=float, default=0.05, help="MIPGap（如 0.05）")
    parser.add_argument("--unit_per_km", type=float, default=5.0, help="坐标单位/公里，例如 5 表示 5单位=1km")
    parser.add_argument("--verbose", type=int, default=1, help="是否打印求解日志(1/0)")
    parser.add_argument("--plot", type=int, default=0, help="是否保存可视化图片(1/0)，需要 matplotlib")
    parser.add_argument("--plot_path", type=str, default="", help="可视化输出 PNG 路径，默认 milp_solution.png")
    parser.add_argument("--plot_show", type=int, default=0, help="是否弹窗显示图像(1/0)，仅本地有界面时有效")
    args = parser.parse_args()

    build_and_solve(
        csv_path=args.csv,
        E_roundtrip_km=args.E,
        truck_speed_kmh=args.truck_speed,
        truck_road_factor=args.truck_road_factor,
        drone_speed_kmh=args.drone_speed,
        alpha=args.alpha,
        lambda_late=args.lambda_late,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        unit_per_km=args.unit_per_km,
        verbose=args.verbose,
        plot=args.plot,
        plot_path=args.plot_path,
        plot_show=args.plot_show,
    )

# ----------------------------
# 直接在文件里改参数点运行（推荐）
# ----------------------------
RUN_WITH_INLINE_CONFIG = True  # 改成 False 就走命令行 argparse

INLINE_CFG = dict(
    csv_path=r"/OR-Tool/25/nodes_25_seed2023_20260110_201842_promise.csv",  # 改成你的数据路径
    E_roundtrip_km=10.0,
    truck_speed_kmh=30.0,
    truck_road_factor=1.5,
    drone_speed_kmh=60.0,
    alpha=0.3,
    lambda_late=50.0,
    time_limit=6000.0,
    mip_gap=0.0,
    unit_per_km=5.0,
    verbose=1,
    plot=1,
    plot_path=r"D:\代码\ALNS+DL\pic-grb\milp_solution.png",
    plot_show=1,
)

def make_plot_path(csv_path: str, out_dir: str, suffix: str = "", ext: str = ".png") -> str:
    """
    根据数据集文件名自动生成不覆盖的图片路径：
    - 从 csv 文件名里提取 nodes/events 的关键信息（如 nodes_25_seed2023...）
    - 加时间戳避免覆盖
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]  # nodes_25_seed2023_xxx
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{stem}{suffix}_{ts}"
    return os.path.join(out_dir, f"milp_{tag}{ext}")

if __name__ == "__main__":
    if RUN_WITH_INLINE_CONFIG:
        out_dir = r"/pic-grb"

        # 自动生成不覆盖的图片名（可额外加 _TL120 或 _gap0p05 之类的后缀）
        suffix = f"_TL{int(INLINE_CFG['time_limit'])}_gap{INLINE_CFG['mip_gap']}"
        INLINE_CFG["plot_path"] = make_plot_path(INLINE_CFG["csv_path"], out_dir, suffix=suffix)

        build_and_solve(**INLINE_CFG)
        print("[OUT] plot_path =", INLINE_CFG["plot_path"])
    else:
        main()

