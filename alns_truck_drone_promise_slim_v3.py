"""
单文件求解器（清理版）：单卡车 + 多无人机 + 客户位置在线变更（不含回放模块）

设计目标：
- 保持原有算法/输出口径不变（关键指标与旧版输出口径兼容）
- 在不引入新依赖的前提下，收敛重复逻辑并清理历史遗留实现

文件结构（分区在代码中用大标题标注）：
1) 全局常量与随机种子
2) 在线扰动日志（保存/回放）
3) 基础工具（距离/时间窗/调试打印）
4) 评估与调度（卡车时刻表、多无人机调度、成本合成）
5) 在线扰动（申请生成/回放/应用）
6) ALNS 算子与主循环
7) 可视化与统计输出
8) 实验入口（单次/套件/静态对比/CLI）

说明：本文件仍保持可直接运行（main()/main_cli()），便于你现有实验脚本与复现流程不改。
"""
# alns_truck__drone（new）.py
import math
import random
import numpy as np
import copy
import os, time, csv
DEBUG_QUICK_FILTER = True
import simulation as sim  # 引入仿真模块
import utils as ut # 工具模块
# 数据读入：强制依赖 data_io_1322.py（不再提供单文件兜底，避免掩盖 import 错误）
from data_io_1322 import read_data, Data, recompute_cost_and_nearest_base
# 引入拆分出去的可视化模块
from viz_utils import visualize_truck_drone, compute_global_xlim_ylim, _normalize_decisions_for_viz
from utils import (set_seed, _derive_promise_nodes_path, _is_promise_nodes_file,
    freeze_existing_promise_windows_inplace, apply_promise_windows_inplace,
    write_promise_nodes_csv, load_events_csv, group_events_by_time,
    map_event_class_to_reloc_type, save_decision_log, print_tw_stats, compute_eta_map, _total_late_against_due, emit_scene_late_logs)

# ========= 数据集 Schema（节点 + 动态事件）=========
# 说明：nodes.csv 仅包含静态节点字段；动态请求流由 events.csv 单独给出（EVENT_TIME, NODE_ID, NEW_X, NEW_Y, EVENT_CLASS）。
CSV_REQUIRED_COLS = [
    "NODE_ID","NODE_TYPE","ORIG_X","ORIG_Y","DEMAND","READY_TIME","DUE_TIME"
]
CSV_NODE_TYPES = {"central", "base", "customer"}
EPS = 1e-9
# 四种算子（动作）
DEBUG = False
# 中文注释：迟到定位开关（只建议临时打开，避免刷屏）
DEBUG_LATE = True
DEBUG_LATE_TOPK = 15
DEBUG_LATE_SCENES = None   # 只看 scene=0；想看全部就改成 None
# 中文注释：重插入诊断开关（仅用于定位“某个迟到客户为什么迟到”）
DEBUG_REINS_DIAG = True
# 中文注释：指定某个客户 idx；None 表示自动选择“当前最迟到的卡车客户”
DEBUG_REINS_CID = None

CFG_A = {
    "NAME": "A_paired_baseline",
    "PAIRING_MODE": "paired",
    "late_hard": 0.12,  # 建议显式写在 cfg 里（要更严就 0.10）
    # 固定两对：等价于你原来“捆绑”的两套组合
    "ALLOWED_PAIRS": [
        ("D_random_route", "R_greedy_then_drone"),
        ("D_worst_route",  "R_regret_then_drone"),
    ],
    "DESTROYS": ["D_random_route", "D_worst_route"],
    "REPAIRS":  ["R_greedy_then_drone", "R_regret_then_drone"],
}

CFG_D = {
    "NAME": "D_full_structured",
    "PAIRING_MODE": "free",
    "late_hard": 0.12,  # 建议显式写在 cfg 里（要更严就 0.10）
    "DESTROYS": ["D_random_route", "D_worst_route", "D_reloc_focus_v2", "D_switch_coverage"],
    "REPAIRS":  ["R_greedy_only", "R_regret_only", "R_greedy_then_drone", "R_regret_then_drone", "R_late_repair_reinsert", "R_base_feasible_drone_first"],
}

def dprint(*args, **kwargs):
    """统一的调试打印开关，避免到处散落 print"""
    if DEBUG:
        print(*args, **kwargs)

# ==============
# 一些基础函数
# ==============
def relax_drone_time_windows(data: Data,
                             base_to_drone_customers,
                             extra_slack: float = 2.0,
                             reset_ready_to_zero: bool = True):
    """
    放宽无人机客户的时间窗：
      - 可选：把 ready_time 统一设为 0（无人机随时可以早到）
      - 把 due_time 在原基础上放宽 extra_slack 小时
    """
    for base_idx, clients in base_to_drone_customers.items():
        for idx in clients:
            node = data.nodes[idx]
            if reset_ready_to_zero:
                node['ready_time'] = 0.0
            node['due_time'] = node.get('due_time', 0.0) + extra_slack

def enforce_force_truck_solution(data, route, b2d):
    """
    强制约束：所有 force_truck=1 的客户必须在卡车路径中，且不得出现在无人机任务中。
    这用于避免 ALNS 算子在搜索过程中把“兜底卡车直送”的客户又转回无人机，导致后续场景丢失。
    """
    forced = [i for i in getattr(data, 'customer_indices', []) if data.nodes[i].get('force_truck', 0) == 1]
    if not forced:
        return route, b2d

    # 1) 从无人机任务里移除
    for b in list(b2d.keys()):
        b2d[b] = [c for c in b2d[b] if c not in forced]

    # 2) 确保在卡车路径里（若缺失则插入到末尾仓库前）
    route_set = set(route)
    missing = [c for c in forced if c not in route_set]
    if not missing:
        return route, b2d

    def extra_cost_insert(r, node_idx, pos):
        a = r[pos - 1]
        b = r[pos]
        return sim.truck_arc_cost(data, a, node_idx) + sim.truck_arc_cost(data, node_idx, b) - sim.truck_arc_cost(data, a, b)

    # 若路径长度不足（异常情况），直接拼接
    if len(route) < 2:
        route = route + missing
        return route, b2d

    for node_idx in missing:
        best_pos = None
        best_delta = float('inf')
        # 插入位置：1..len(route)-1，避免插在起点之前
        for pos in range(1, len(route)):
            dlt = extra_cost_insert(route, node_idx, pos)
            if dlt < best_delta:
                best_delta = dlt
                best_pos = pos
        route.insert(best_pos, node_idx)

    return route, b2d

def feasible_bases_for_customer(data, cid, ctx, route_set, drone_range=None):
    """返回客户 cid 在当前上下文下可用的基站集合。

    语义区分：
    - ctx['bases_to_visit']：未来将访问的基站（用于路径规划/业务“换基站”逻辑）
    - ctx['visited_bases']：已访问过的基站（arrival_prefix 中的基站）
    - ctx['feasible_bases_for_drone']：无人机可用基站（通常=visited_bases ∪ bases_to_visit）

    可用判定：
    - 覆盖：2*d(base, client) <= drone_range
    - 供货：基站要么已访问（visited_bases），要么在后缀路线中会访问（route_set）
    """
    if drone_range is None:
        drone_range = sim.DRONE_RANGE_UNITS  # 从 sim 获取最新值
    feas_pool = ctx.get('feasible_bases_for_drone', None)
    if feas_pool is None:
        feas_pool = ctx.get('bases_to_visit', [])

    visited = set(ctx.get('visited_bases', []))

    feas = []
    x, y = float(data.nodes[cid]['x']), float(data.nodes[cid]['y'])
    for b in feas_pool:
        if b is None:
            continue
        b = int(b)
        # 供货条件
        if (b not in route_set) and (b not in visited):
            continue
        bx, by = float(data.nodes[b]['x']), float(data.nodes[b]['y'])
        d = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
        if 2.0 * d <= float(drone_range) + 1e-9:
            feas.append(b)
    return feas
def drone_repair_feasible(data, route, b2d, ctx, k_moves=5, sample_k=12):
    """
    在约束下（bases_to_visit/覆盖圈/base_lock/force_truck）对 b2d 做局部改进：
    - truck -> drone
    - drone -> truck
    - drone 换基站
    每一步用 evaluate_truck_drone_with_time 真实评估 cost，确保“扎实可靠”。
    """
    bases_to_visit = ctx.get("bases_to_visit", [])
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)
    alpha_drone = ctx.get("alpha_drone", 0.3)
    lambda_late = ctx.get("lambda_late", 50.0)
    truck_speed = ctx.get("truck_speed", sim.TRUCK_SPEED_UNITS)
    drone_speed = ctx.get("drone_speed", sim.DRONE_SPEED_UNITS)
    start_time = ctx.get("start_time", 0.0)

    route_set = set(route)
    force_truck_set = set(ctx.get("force_truck_set", set()))

    # 当前客户在哪
    in_route = {i for i in route if data.nodes[i].get("node_type") == "customer"}
    in_drone = {c for cs in b2d.values() for c in cs}

    candidates = list((in_route | in_drone) - force_truck_set)

    def eval_cost(r, bd):
        return sim.evaluate_truck_drone_with_time(
            data, r, bd,
            alpha_drone=alpha_drone, lambda_late=lambda_late,
            truck_speed=truck_speed, drone_speed=drone_speed,
            start_time=start_time,
            arrival_prefix=ctx.get("arrival_prefix")
        )[0]

    cur_cost = eval_cost(route, b2d)

    for _ in range(k_moves):
        if not candidates:
            break

        pool = random.sample(candidates, min(sample_k, len(candidates)))

        best_delta = 0.0
        best_sol = None  # (new_route, new_b2d, new_cost)

        for cid in pool:
            node = data.nodes[cid]
            locked_b = node.get("base_lock", None)

            # ---- Move 1: truck -> drone ----
            if cid in in_route:
                feas_bases = feasible_bases_for_customer(
                    data, cid, ctx, route_set, drone_range
                )
                if locked_b is not None:
                    feas_bases = [b for b in feas_bases if b == locked_b]

                for b in feas_bases:
                    r2 = [x for x in route if x != cid]
                    bd2 = {bb: lst[:] for bb, lst in b2d.items()}
                    bd2.setdefault(b, [])
                    if cid not in bd2[b]:
                        bd2[b].append(cid)

                    new_cost = eval_cost(r2, bd2)
                    delta = new_cost - cur_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_sol = (r2, bd2, new_cost)

            # ---- Move 2: drone -> truck ----
            if cid in in_drone:
                bd2 = {bb: [c for c in lst if c != cid] for bb, lst in b2d.items()}
                r2 = greedy_insert(data, route, [cid])
                new_cost = eval_cost(r2, bd2)
                delta = new_cost - cur_cost
                if delta < best_delta:
                    best_delta = delta
                    best_sol = (r2, bd2, new_cost)

            # ---- Move 3: drone 换基站 ----
            if cid in in_drone:
                cur_b = None
                for bb, lst in b2d.items():
                    if cid in lst:
                        cur_b = bb
                        break
                if cur_b is not None:
                    feas_bases = feasible_bases_for_customer(
                        data, cid, ctx, route_set, drone_range
                    )
                    if locked_b is not None:
                        feas_bases = [b for b in feas_bases if b == locked_b]

                    for b in feas_bases:
                        if b == cur_b:
                            continue
                        bd2 = {bb: lst[:] for bb, lst in b2d.items()}
                        bd2[cur_b] = [c for c in bd2[cur_b] if c != cid]
                        bd2.setdefault(b, [])
                        bd2[b].append(cid)

                        new_cost = eval_cost(route, bd2)
                        delta = new_cost - cur_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_sol = (route, bd2, new_cost)

        if best_sol is None:
            break

        route, b2d, cur_cost = best_sol
        route_set = set(route)
        in_route = {i for i in route if data.nodes[i].get("node_type") == "customer"}
        in_drone = {c for cs in b2d.values() for c in cs}
        candidates = list((in_route | in_drone) - force_truck_set)

    return route, b2d

def pick_worst_late_truck_customer(data, route, full_eval, eps: float = 1e-9):
    """中文注释：从当前解里自动找出“卡车服务客户”中迟到最大的一个。
    返回：(idx, late)。若没有卡车客户迟到，则返回 (None, 0.0)。
    说明：这里用 full_eval 里的 arrival（卡车到达/服务时刻）与数据中的 due_time 计算迟到。
    """
    arrival = full_eval.get("arrival", {}) or {}
    worst_idx, worst_late = None, 0.0

    for idx in route:
        # 只看客户节点（基站/中心仓库不参与“客户迟到”）
        nt = str(data.nodes[idx].get("node_type", data.nodes[idx].get("type", ""))).strip().lower()
        if not ((nt == "customer") or (nt == "c") or ("cust" in nt)):
            continue
        if idx not in arrival:
            continue
        due = float(data.nodes[idx].get("due_time", float("inf")))
        if due == float("inf"):
            continue
        late = max(0.0, float(arrival[idx]) - due)
        if late > worst_late + eps:
            worst_late, worst_idx = late, idx

    return worst_idx, float(worst_late)

def build_client_to_base_map(base_to_drone_customers):
    """
    把 {base_idx: [client_idx,...]} 反转成 {client_idx: base_idx}
    方便后面判断某个客户是不是无人机客户，以及所属基站。
    """
    mapping = {}
    for b, clist in base_to_drone_customers.items():
        for c in clist:
            mapping[c] = b
    return mapping

def apply_relocations_for_decision_time(
    data, t_prev, decision_time,
    depart_times, finish_times, arrival_times, client_to_base,
    req_clients_override, predefined_xy, predefined_types=None, predefined_delta_avail=None
):
    """
    在给定决策时刻，将提出位置变更的客户按照规则处理：
      - 无人机客户：同基站内且无人机尚未起飞 → 更新为扰动坐标
      - 卡车客户：
          * 若卡车已服务该客户 → 拒绝
          * 否则看新位置：
              - 若在任意基站圈外 → 仍为卡车客户，只更新坐标
              - 若在某基站圈内且卡车未来仍会经过该基站 → 接受，更新坐标（后续由分类函数决定是否转为无人机客户）
              - 若在基站圈内但卡车已路过该基站 → 拒绝
    返回：
      - data_new: 更新后的 Data（深拷贝）
      - decisions: 决策列表，方便打印
    """
    data_new = copy.deepcopy(data)
    decisions = []  # (client_idx, node_id, decision, reason, old_x, old_y, new_x, new_y)
    # ---------- 强制检查：arrival_times 必须是从 t=0 开始的全局时间轴 ----------
    t0 = arrival_times.get(data_new.central_idx, None)
    if t0 is None:
        raise RuntimeError("arrival_times 缺少 central_idx 的到达时刻，必须传 full_arrival_cur")
    if abs(t0 - 0.0) > 1e-6:
        raise RuntimeError(f"arrival_times 不是全局时间轴：arrival[central]={t0}，必须传 full_arrival_cur（从0开始）")

    if not (t_prev < decision_time + 1e-9):
        raise RuntimeError(f"决策窗口非法：t_prev={t_prev}, t_dec={decision_time}")

    # 预先算出所有“基站索引”（含中心仓库）
    base_indices = [i for i, n in enumerate(data_new.nodes)
                    if n['node_type'] == 'base']
    if data_new.central_idx not in base_indices:
        base_indices.insert(0, data_new.central_idx)

    # 当前时刻的变更请求客户（events-only：请求流由 events.csv 提供）
    # 说明：本函数只负责把 NEW_X/NEW_Y 写入节点的临时字段（perturbed_x/perturbed_y），
    #      是否真正更新为当前坐标（current x/y）由下方 ACCEPT/REJECT 规则决定。
    if req_clients_override is None:
        raise RuntimeError("events-only 模式：必须传入 req_clients_override（来自 events.csv 的 NODE_ID 列）")
    if predefined_xy is None:
        raise RuntimeError("events-only 模式：必须传入 predefined_xy（来自 events.csv 的 NEW_X/NEW_Y 列）")

    req_clients = [int(c) for c in req_clients_override]

    # 将本次决策点请求写入临时字段（不做边界裁剪，保证事实坐标不被篡改）
    for c in req_clients:
        if 0 <= c < len(data_new.nodes):
            node = data_new.nodes[c]
            if str(node.get("node_type", "")).lower() != "customer":
                continue
            if int(c) not in predefined_xy:
                raise RuntimeError(f"events-only 模式：客户 idx={c} 缺少 NEW_X/NEW_Y（predefined_xy 未包含该客户）")
            nx, ny = predefined_xy[int(c)]
            node["perturbed_x"] = float(nx)
            node["perturbed_y"] = float(ny)
            node["request_time"] = float(decision_time)
            if predefined_types is not None and int(c) in predefined_types:
                node["reloc_type"] = str(predefined_types[int(c)])            # 中文注释：事件的相对可用时长（小时），用于后续在求解端计算 L 与有效截止时间
            if predefined_delta_avail is not None and int(c) in predefined_delta_avail:
                node["delta_avail_h"] = float(predefined_delta_avail[int(c)])
            else:
                node["delta_avail_h"] = float(node.get("delta_avail_h", 0.0))
            # 中文注释：冻结承诺窗（若缺失则回退到现有 ready/due）
            if "prom_ready" not in node:
                node["prom_ready"] = float(node.get("ready_time", 0.0))
            if "prom_due" not in node:
                node["prom_due"] = float(node.get("due_time", node.get("prom_ready", 0.0)))
            # 中文注释：候选有效截止时间 L（仅当最终接受且未服务时才生效）
            node["candidate_effective_due"] = max(float(node["prom_due"]), float(node["prom_ready"]) + float(node["delta_avail_h"]))
    for c in req_clients:
        node = data_new.nodes[c]
        nid = node['node_id']
        old_x = node.get('x')
        old_y = node.get('y')

        # 先取扰动坐标
        px = node.get('perturbed_x')
        py = node.get('perturbed_y')

        # 如果扰动坐标无效，直接拒绝
        if px is None or py is None or math.isnan(px) or math.isnan(py):
            decisions.append((c, nid, "REJECT",
                              "无有效扰动坐标(PERTURBED_X/Y 为空或 NaN)",
                              old_x, old_y, old_x, old_y))
            continue

        # ---------- 1) 无人机客户：允许“货还在卡车上时换基站” ----------
        if c in client_to_base:
            b_old = client_to_base[c]  # 旧基站 idx（不是 node_id）

            # (U0) 已完成：拒绝
            t_fin = finish_times.get(c, float('inf'))
            if t_fin <= decision_time + 1e-9:
                decisions.append((c, nid, "REJECT",
                                  f"无人机客户：已在 t={t_fin:.2f}h 完成服务，t_dec={decision_time:.2f}h",
                                  old_x, old_y, px, py))
                continue

            # (U1) 已起飞/正在执行：拒绝
            t_dep = depart_times.get(c, float('inf'))
            if t_dep <= decision_time + 1e-9:
                decisions.append((c, nid, "REJECT",
                                  f"无人机客户：已/正在起飞 (t_depart={t_dep:.2f}h)，t_dec={decision_time:.2f}h",
                                  old_x, old_y, px, py))
                continue

            # 货是否已到旧基站（=卡车是否已到/路过旧基站）
            t_old_base = arrival_times.get(b_old, float('inf'))

            # 计算“新位置到某基站是否可覆盖”（往返 <= DRONE_RANGE）
            def in_cover(bidx):
                bx = data_new.nodes[bidx]['x']
                by = data_new.nodes[bidx]['y']
                d = math.hypot(px - bx, py - by)
                return (2.0 * d <= sim.DRONE_RANGE_UNITS), d

            # 情况A：货已到旧基站 -> 只能同基站内变更（不允许换基站）
            if t_old_base <= decision_time + 1e-9:
                ok, d_old = in_cover(b_old)
                if not ok:
                    base_nid = data_new.nodes[b_old]['node_id']
                    decisions.append((c, nid, "REJECT",
                                      f"无人机客户：货已到旧基站 node_id={base_nid} (t={t_old_base:.2f}h)，"
                                      f"但新位置超出该基站覆盖范围，拒绝",
                                      old_x, old_y, px, py))
                    continue

                _apply_xy(node, px, py, source="PERTURBED")
                node['base_lock'] = b_old  # 可选：锁回原基站
                decisions.append((c, nid, "ACCEPT",
                                  "无人机客户：货已在基站，且新位置仍在同基站覆盖范围内，接受并更新坐标",
                                  old_x, old_y, px, py))
                continue

            # 情况B：货还在卡车上（卡车未到旧基站） -> 允许换到“未来会经过且能覆盖”的基站
            # 未来可达基站集合：arrival[b] > t_dec
            base_indices_all = [i for i, n in enumerate(data_new.nodes) if n['node_type'] == 'base']
            if data_new.central_idx not in base_indices_all:
                base_indices_all.insert(0, data_new.central_idx)

            future_bases = []
            for bidx in base_indices_all:
                t_b = arrival_times.get(bidx, float('inf'))
                if t_b <= decision_time + 1e-9:
                    continue  # 已经路过的不考虑
                ok, d = in_cover(bidx)
                if ok:
                    future_bases.append((d, bidx, t_b))

            if not future_bases:
                # 兜底：你更“客户友好”的版本 -> 接受，但强制卡车直送
                _apply_xy(node, px, py, source="PERTURBED")

                node['force_truck'] = 1
                node['base_lock'] = None
                decisions.append((c, nid, "ACCEPT",
                                  "无人机客户：货还在卡车上，但新位置不被任何未来基站覆盖；改为卡车直送兜底并更新坐标（force_truck=1）",
                                  old_x, old_y, px, py))
                continue

            # 选择最近的“未来可达且可覆盖”的基站
            d_best, b_best, t_best = min(future_bases, key=lambda x: x[0])

            _apply_xy(node, px, py, source="PERTURBED")

            node['base_lock'] = b_best  # ✅ 锁定到新基站
            node['force_truck'] = 0  # ✅ 仍允许无人机模式（后续分类会尊重 base_lock）
            base_nid = data_new.nodes[b_best]['node_id']

            decisions.append((c, nid, "ACCEPT",
                              f"无人机客户：货还在卡车上（未到旧基站），新位置由未来基站 node_id={base_nid} 覆盖，"
                              f"且卡车未来将于 t={t_best:.2f}h 到达该基站；接受并切换基站",
                              old_x, old_y, px, py))
            continue

            # 同基站内更新坐标
            _apply_xy(node, px, py, source="PERTURBED")

            decisions.append((c, nid, "ACCEPT",
                              "无人机客户：同一基站内，决策时刻早于无人机起飞，更新为扰动坐标",
                              old_x, old_y, px, py))
            continue

        # ---------- 2) 卡车客户：先看是否已经被服务 ----------
        t_arr = arrival_times.get(c, float('inf'))
        if decision_time >= t_arr:
            decisions.append((c, nid, "REJECT",
                              f"卡车已在 t={t_arr:.2f}h 服务该客户，t_dec={decision_time:.2f}h",
                              old_x, old_y, px, py))
            continue

        # ---------- 3) 判断新位置的覆盖基站集合（考虑重叠覆盖） ----------
        cover_bases = []  # [(dist, bidx), ...]
        for bidx in base_indices:
            bx = data_new.nodes[bidx]['x']
            by = data_new.nodes[bidx]['y']
            d = math.hypot(px - bx, py - by)

            # 覆盖判断：往返 <= DRONE_RANGE_UNITS
            if 2.0 * d <= DRONE_RANGE_UNITS:
                cover_bases.append((d, bidx))

        # 不在任何覆盖圈：仍由卡车服务，更新坐标
        if not cover_bases:
            _apply_xy(node, px, py, source="PERTURBED")

            decisions.append((c, nid, "ACCEPT",
                              "卡车客户：新位置不在任何基站圈内，仍由卡车服务，但更新坐标",
                              old_x, old_y, px, py))
        else:
            # 在覆盖圈内：必须存在一个“未来还能到达”的基站，否则拒绝
            future_bases = []
            for d, bidx in cover_bases:
                t_base = arrival_times.get(bidx, float('inf'))
                if t_base > decision_time + 1e-9:
                    future_bases.append((d, bidx, t_base))

            if not future_bases:
                # ✅ 兜底：所有覆盖基站都已路过，但仍允许“卡车直送”并更新坐标
                _apply_xy(node, px, py, source="PERTURBED")

                node['force_truck'] = 1
                node['base_lock'] = None  # 这里不锁基站，后续也不要转无人机

                # 为了可解释性，记录一个最近覆盖基站的信息（仅用于打印原因）
                d0, b0 = min(cover_bases, key=lambda x: x[0])
                t0 = arrival_times.get(b0, float('inf'))
                base_nid = data_new.nodes[b0]['node_id']

                decisions.append((c, nid, "ACCEPT",
                                  f"卡车客户：新位置落在基站覆盖圈内，但所有可覆盖基站均已被卡车路过"
                                  f"（例如最近基站 node_id={base_nid} 于 t={t0:.2f}h）；"
                                  f"为保证客户友好性，改为卡车直送兜底并更新坐标（force_truck=1），t_dec={decision_time:.2f}h",
                                  old_x, old_y, px, py))
            else:
                # 选择“最近且未来可达”的基站 -> 可潜在转无人机
                d_best, b_best, t_best = min(future_bases, key=lambda x: x[0])
                _apply_xy(node, px, py, source="PERTURBED")

                node['force_truck'] = 0
                node['base_lock'] = b_best  # 可选：锁到该基站，避免后续又分配到别的基站
                base_nid = data_new.nodes[b_best]['node_id']
                decisions.append((c, nid, "ACCEPT",
                                  f"卡车客户：新位置在未来基站 node_id={base_nid} 覆盖圈内，且卡车未来将于 t={t_best:.2f}h 到达该基站；接受并更新坐标",
                                  old_x, old_y, px, py))

    # 4) 所有更新完成后，重算距离矩阵和最近基站
    #    重要修复：这里不能只在“未来可达基站”里选最近。
    #    否则当卡车已经路过所有基站时（future bases 为空），会导致 recompute 直接报错。
    #    base_id 的重算应基于“全体基站(含 central)”，后续真正的可行基站约束在
    #    classify_clients_for_drone / ALNS 的 bases_to_visit 中处理。
    base_indices_all = [i for i, n in enumerate(data_new.nodes) if n['node_type'] == 'base']
    if data_new.central_idx not in base_indices_all:
        base_indices_all.insert(0, data_new.central_idx)
    
    future_bases = [b for b in base_indices_all
                    if arrival_times.get(b, float('inf')) > decision_time + 1e-9]
    dprint("[debug] future_bases(node_id)=", [data_new.nodes[b]['node_id'] for b in future_bases])
    
    # 这里传 None => 使用全部基站计算最近基站；不会因 future_bases 为空而崩溃
    recompute_cost_and_nearest_base(data_new, feasible_bases=None)

    return data_new, decisions, req_clients

def split_route_by_decision_time(route, arrival_times, decision_time, central_idx, data):
    """
    用“全局 arrival_times”在 route 上做切分：
      - served_nodes / remaining_nodes：按节点是否已到达判断（不含 depot）
      - current_node / virtual_pos：决策时刻在节点上或在边上（线性插值）
      - prefix_route：从 route[0] 到“决策时刻所在边的前一节点 prev_idx”为止的真实前缀（按 route 顺序）
        这是后续拼 full path 的关键，用它可保证决策前前缀冻结不变。
    """
    if route is None or len(route) < 2:
        return [], [], central_idx, None, [central_idx]

    served_nodes = []
    remaining_nodes = []

    current_node = central_idx
    virtual_pos = None

    # prefix_end_pos：前缀最后一个节点在 route 中的位置（包含该节点）
    prefix_end_pos = 0

    # arrival of first node（通常是 depot）
    prev_idx = route[0]
    prev_t = arrival_times.get(prev_idx, 0.0)

    # 逐段找 t_dec 落在哪儿
    for pos in range(1, len(route)):
        idx = route[pos]
        t_arr = arrival_times.get(idx, float('inf'))

        if t_arr <= decision_time + 1e-9:
            # 决策时刻已经“到达” idx
            if idx != central_idx and data.nodes[idx]['node_type'] != 'base':
                served_nodes.append(idx)
            current_node = idx
            virtual_pos = None

            prefix_end_pos = pos
            prev_idx = idx
            prev_t = t_arr
        else:
            # 决策时刻在 [prev_idx -> idx] 之间（或仍停在 prev_idx）
            # 这里沿用你之前的做法：直接用 (arrival[prev], arrival[idx]) 做插值近似
            if decision_time > prev_t + 1e-9 and t_arr > prev_t + 1e-9:
                ratio = (decision_time - prev_t) / (t_arr - prev_t)
                ratio = max(0.0, min(1.0, ratio))
                x_prev = data.nodes[prev_idx]['x']
                y_prev = data.nodes[prev_idx]['y']
                x_idx = data.nodes[idx]['x']
                y_idx = data.nodes[idx]['y']
                x_cur = x_prev + ratio * (x_idx - x_prev)
                y_cur = y_prev + ratio * (y_idx - y_prev)
                virtual_pos = (x_cur, y_cur)
                current_node = None
            else:
                # 决策时刻 <= prev_t，认为车还在 prev_idx
                current_node = prev_idx
                virtual_pos = None

            # 剩余节点：从 idx 开始的后续（去掉 depot/base）
            for k in route[pos:]:
                if k == central_idx:
                    continue
                if data.nodes[k]['node_type'] == 'base':
                    continue
                remaining_nodes.append(k)

            break

    prefix_route = route[:prefix_end_pos + 1]
    # 确保前缀至少包含 depot
    if not prefix_route:
        prefix_route = [central_idx]

    return served_nodes, remaining_nodes, current_node, virtual_pos, prefix_route

def add_virtual_truck_position_node(data, pos):
    """
    在 data 中新增一个表示“当前卡车位置”的虚拟节点，并返回其索引。
    pos: (x, y)
    """
    if pos is None:
        return None

    x_cur, y_cur = pos
    new_idx = len(data.nodes)

    data.nodes.append({
        "node_id": -1,           # 虚拟点，node_id 给个特殊值
        "node_type": "truck_pos",
        "x": x_cur,
        "y": y_cur,
        "ready_time": 0.0,
        "due_time": 1e9
    })

    n_old = data.costMatrix.shape[0]
    new_mat = np.zeros((n_old + 1, n_old + 1))
    new_mat[:n_old, :n_old] = data.costMatrix

    for j in range(n_old):
        xj = data.nodes[j]['x']
        yj = data.nodes[j]['y']
        d = math.hypot(x_cur - xj, y_cur - yj)
        new_mat[new_idx, j] = d
        new_mat[j, new_idx] = d

    data.costMatrix = new_mat
    return new_idx

def _pack_scene_record(scene_idx, t_dec, full_eval, num_req, num_acc, num_rej,
                       alpha_drone=0.3, lambda_late=50.0):
    """统一封装动态场景记录行，确保关键指标口径一致。"""
    base_cost = float(full_eval.get("truck_dist_eff", full_eval["truck_dist"]))                 + float(alpha_drone) * float(full_eval["drone_dist"])
    penalty = float(lambda_late) * float(full_eval["total_late"])
    return {
        "scene": int(scene_idx),
        "t_dec": float(t_dec),
        "cost": float(full_eval["cost"]),
        "base_cost": base_cost,
        "penalty": penalty,
        "lambda_late": float(lambda_late),
        "truck_dist": float(full_eval["truck_dist"]),
        "drone_dist": float(full_eval["drone_dist"]),
        "system_time": float(full_eval["system_time"]),
        "truck_late": float(full_eval["truck_late"]),
        "drone_late": float(full_eval["drone_late"]),
        "total_late": float(full_eval["total_late"]),
        "num_req": int(num_req),
        "num_acc": int(num_acc),
        "num_rej": int(num_rej),
    }


def _decision_tag(decision_item):
    """从 decisions 的元素中提取 'ACCEPT...' 或 'REJECT...' 标记（兼容历史格式）。"""
    if isinstance(decision_item, (list, tuple)):
        for x in decision_item:
            if isinstance(x, str) and (x.startswith("ACCEPT") or x.startswith("REJECT")):
                return x
    return ""




def _merge_prefix_suffix(prefix_route, suffix_route):
    """将已执行前缀与新求解后缀拼接为全局路线（去重连接点）。"""
    if not prefix_route:
        return suffix_route[:]
    if not suffix_route:
        return prefix_route[:]
    if prefix_route[-1] == suffix_route[0]:
        return prefix_route + suffix_route[1:]
    return prefix_route + suffix_route

def nearest_neighbor_route_truck_only(data: Data,
                                      truck_customers,
                                      start_idx=None,
                                      end_idx=None,
                                      bases_to_visit=None):

    """
    只在：中心仓库 + 所有基站 + 需要卡车服务的客户 上做 TSP。
    route 是这些节点的全局索引序列。

    start_idx: 卡车出发节点（动态场景用 current_pos），默认中心仓库
    end_idx  : 最终返回节点，默认中心仓库
    """
    # 起点、终点默认都是中心仓库
    if start_idx is None:
        start_idx = data.central_idx
    if end_idx is None:
        end_idx = data.central_idx

    if bases_to_visit is None:
        # 静态默认：所有基站
        bases_to_visit = [i for i, n in enumerate(data.nodes) if n.get('node_type') == 'base']
        if data.central_idx not in bases_to_visit:
            bases_to_visit.append(data.central_idx)

    allowed = set(bases_to_visit) | set(truck_customers) | {start_idx, end_idx}

    # 终点只在最后附加，不作为中途访问节点
    if end_idx in allowed:
        allowed.remove(end_idx)
    # 起点视为已访问
    if start_idx in allowed:
        allowed.remove(start_idx)

    route = [start_idx]
    current = start_idx

    unvisited = allowed.copy()
    while unvisited:
        next_node = min(unvisited, key=lambda j: sim.truck_arc_cost(data, current, j))
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    # 最后回到终点
    if route[-1] != end_idx:
        route.append(end_idx)

    return route


def random_removal(route, num_remove, data: Data, protected=None):
    """
    大邻域中的 '破坏算子'：
    - 从当前 route 中随机移除 num_remove 个非起终点节点
    - 返回: (new_route, removed_nodes)
    """
    if protected is None:
        protected = set()

    if num_remove <= 0:
        return route[:], []

    inner_nodes = [x for x in route[1:-1] if x not in protected]
    # 不移除首尾（中心仓库）
    if num_remove >= len(inner_nodes):
        num_remove = len(inner_nodes)

    removed = random.sample(inner_nodes, num_remove)
    remaining_inner = [i for i in route[1:-1] if i not in removed]
    remaining = [route[0]] + remaining_inner + [route[-1]]
    return remaining, removed

def worst_removal(route, num_remove, data: Data, protected=None):
    """
    删除对当前卡车距离贡献最大的若干节点
    route: 当前完整路径 [0, ..., 0]
    num_remove: 要删除的节点数量（不包含首尾仓库）
    返回: destroyed_route, removed_nodes
    """
    if protected is None:
        protected = set()
    # 内部节点位置（不含首尾）
    inner_positions = list(range(1, len(route) - 1))
    if not inner_positions:
        return route[:], []

    # 计算每个节点的“贡献”：删除它能减少多少卡车距离
    contributions = []
    for pos in inner_positions:
        i = route[pos]
        if i in protected:
            continue
        a = route[pos - 1]
        b = route[pos + 1]
        saving = (sim.truck_arc_cost(data, a, i) +
                  sim.truck_arc_cost(data, i, b) -
                  sim.truck_arc_cost(data, a, b))
        # saving 越大，说明删掉它越有利
        contributions.append((saving, pos, i))

    # 按 saving 从大到小排序，选前 num_remove 个
    contributions.sort(reverse=True, key=lambda x: x[0])
    to_remove = [pos for (_, pos, _) in contributions[:num_remove]]

    # 构造删除后的路径
    to_remove_set = set(to_remove)
    destroyed_route = [node for idx, node in enumerate(route) if idx not in to_remove_set]
    removed_nodes = [route[pos] for pos in to_remove]

    return destroyed_route, removed_nodes

def greedy_insert(data: Data, route, removed_nodes):
    """
    大邻域中的 '修复算子'：
    - 将 removed_nodes 贪心地插回 route 中
    - 每次选择 插入位置 使得增量成本最小
    """
    new_route = route[:]

    for node in removed_nodes:
        best_pos = None
        best_delta = float('inf')

        # 在 [0, 1, ..., len(new_route)-1] 之间找插入位置 i
        # 表示插入到 new_route[i] 和 new_route[i+1] 之间
        for i in range(len(new_route) - 1):
            a = new_route[i]
            b = new_route[i + 1]
            old_cost = sim.truck_arc_cost(data, a, b)
            new_cost = sim.truck_arc_cost(data, a, node) + sim.truck_arc_cost(data, node, b)
            delta = new_cost - old_cost

            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1

        new_route.insert(best_pos, node)

    return new_route

def regret_insert(data: Data, destroyed_route, removed_nodes):
    """
    Regret-2 插入：
      - 对每个未插入客户 k：
          枚举所有插入位置 pos，计算增量成本 delta
          取最小增量 best1 和第二小增量 best2
          regret(k) = best2 - best1
      - 每次选择 regret 最大的客户，插入其 best1 位置
    """
    route = destroyed_route[:]

    if not removed_nodes:
        return route

    while removed_nodes:
        best_k = None
        best_pos = None
        best_delta = None
        best_regret = -1e9

        # 对每个待插入客户，计算 regret
        for k in removed_nodes:
            deltas = []
            # 枚举插入位置：在 route[pos-1] 和 route[pos] 之间插入 k
            for pos in range(1, len(route)):  # 不在起点前插
                a = route[pos - 1]
                b = route[pos]
                delta = (sim.truck_arc_cost(data, a, k) +
                         sim.truck_arc_cost(data, k, b) -
                         sim.truck_arc_cost(data, a, b))
                deltas.append((delta, pos))

            # 按增量成本排序
            deltas.sort(key=lambda x: x[0])
            best1_delta, best1_pos = deltas[0]
            if len(deltas) > 1:
                best2_delta = deltas[1][0]
            else:
                best2_delta = best1_delta

            regret = best2_delta - best1_delta

            # 选择 regret 最大的客户
            if regret > best_regret:
                best_regret = regret
                best_k = k
                best_pos = best1_pos
                best_delta = best1_delta

        # 把 best_k 插入 best_pos
        route.insert(best_pos, best_k)
        removed_nodes.remove(best_k)

    return route

def D_random_route(data, route, b2d, ctx):
    num_remove = ctx["num_remove"]
    protected = ctx.get("protected_nodes", set())
    destroyed_route, removed = random_removal(route, num_remove, data, protected=protected)
    destroyed_b2d = {b: lst[:] for b, lst in b2d.items()}  # 先不动无人机
    return destroyed_route, destroyed_b2d, removed

def D_worst_route(data, route, b2d, ctx):
    num_remove = ctx["num_remove"]
    protected = ctx.get("protected_nodes", set())
    destroyed_route, removed = worst_removal(route, num_remove, data, protected=protected)
    destroyed_b2d = {b: lst[:] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def D_reloc_focus_v2(data, route, b2d, ctx):
    num_remove = int(ctx.get("num_remove", 1))
    protected = set(ctx.get("protected_nodes", set()))

    moved_acc = set(ctx.get("C_moved_accept", set()))
    moved_rej = set(ctx.get("C_moved_reject", set()))
    force_set = set(ctx.get("C_force_truck", set()))
    boundary = set(ctx.get("C_boundary", set()))

    # 基础池：acc/force/boundary 全权重；rej 低权重
    pool_main = (moved_acc | force_set | boundary)
    pool_rej = moved_rej - pool_main

    def is_removable(i):
        return (0 <= i < len(data.nodes)
                and data.nodes[i].get("node_type") == "customer"
                and i not in protected)

    pool_main = [i for i in pool_main if is_removable(i)]
    pool_rej = [i for i in pool_rej if is_removable(i)]

    removed = []

    # 1) 先从 main 抽
    if pool_main:
        take = min(num_remove, len(pool_main))
        removed.extend(random.sample(pool_main, take))

    # 2) 不够再从 rej 抽（降权：最多补 1/3）
    if len(removed) < num_remove and pool_rej:
        need = num_remove - len(removed)
        cap = max(1, num_remove // 3)
        take = min(need, cap, len(pool_rej))
        removed.extend(random.sample(pool_rej, take))

    # 3) 连带破坏：在卡车路径中找 removed 的邻居（前驱/后继）
    route_pos = {node: idx for idx, node in enumerate(route)}
    extra = []
    for c in list(removed):
        if c not in route_pos:
            continue
        j = route_pos[c]
        for nb in [route[j-1] if j-1 >= 0 else None, route[j+1] if j+1 < len(route) else None]:
            if nb is None:
                continue
            if is_removable(nb) and nb not in removed and nb not in extra:
                extra.append(nb)
    # 只加到够 num_remove 为止
    random.shuffle(extra)
    for nb in extra:
        if len(removed) >= num_remove:
            break
        removed.append(nb)

    # 4) 仍不足：从 route 内部补随机
    if len(removed) < num_remove:
        inner = [i for i in route[1:-1] if is_removable(i) and i not in removed]
        need = num_remove - len(removed)
        if inner:
            removed += random.sample(inner, min(need, len(inner)))

    removed_set = set(removed)

    destroyed_route = [x for x in route if x not in removed_set]
    destroyed_b2d = {b: [c for c in lst if c not in removed_set] for b, lst in b2d.items()}

    return destroyed_route, destroyed_b2d, removed

def D_switch_coverage(data, route, b2d, ctx):
    num_remove = int(ctx.get("num_remove", 1))
    protected = set(ctx.get("protected_nodes", set()))
    bases_to_visit = ctx.get("bases_to_visit", [])
    drone_range = ctx.get("drone_range", None)
    if drone_range is None:
        return route[:], {b: lst[:] for b, lst in b2d.items()}, []

    route_set = set(route)
    force_set = set(ctx.get("force_truck_set", set()))

    in_route = {i for i in route if data.nodes[i].get("node_type") == "customer"}
    in_drone = {c for cs in b2d.values() for c in cs}
    cand = list((in_route | in_drone) - force_set)

    def ok(i):
        if i in protected:
            return False
        if data.nodes[i].get("node_type") != "customer":
            return False
        feas = feasible_bases_for_customer(data, i, ctx, route_set, drone_range)
        # base_lock 若存在，只能锁定基站可行才算可切换
        lockb = data.nodes[i].get("base_lock", None)
        if lockb is not None:
            feas = [b for b in feas if b == lockb]
        return len(feas) > 0

    cand = [i for i in cand if ok(i)]
    if not cand:
        return route[:], {b: lst[:] for b, lst in b2d.items()}, []

    removed = random.sample(cand, min(num_remove, len(cand)))
    removed_set = set(removed)

    destroyed_route = [x for x in route if x not in removed_set]
    destroyed_b2d = {b: [c for c in lst if c not in removed_set] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def R_greedy_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r = greedy_insert(data, destroyed_route, removed_customers)
    bd = {b: lst[:] for b, lst in destroyed_b2d.items()}
    return r, bd

def R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r = regret_insert(data, destroyed_route, removed_customers)
    bd = {b: lst[:] for b, lst in destroyed_b2d.items()}
    return r, bd

def R_greedy_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r, bd = R_greedy_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    return r, bd

def R_regret_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r, bd = R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    return r, bd

def R_base_feasible_drone_first(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    bases_to_visit = ctx.get("bases_to_visit", [])
    drone_range = ctx.get("drone_range", None)
    if drone_range is None:
        # 兜底：全插回卡车
        r = greedy_insert(data, destroyed_route, removed_customers)
        bd = {b: lst[:] for b, lst in destroyed_b2d.items()}
        return r, bd

    route = destroyed_route[:]
    b2d = {b: lst[:] for b, lst in destroyed_b2d.items()}
    route_set = set(route)
    force_set = set(ctx.get("force_truck_set", set()))

    # 先无人机分配
    for cid in removed_customers:
        if cid in force_set:
            continue
        feas = feasible_bases_for_customer(data, cid, ctx, route_set, drone_range)
        lockb = data.nodes[cid].get("base_lock", None)
        if lockb is not None:
            feas = [b for b in feas if b == lockb]
        if feas:
            b = random.choice(feas)  # 先随机，后面可改成按增量成本最小选
            b2d.setdefault(b, [])
            b2d[b].append(cid)
        else:
            # 无法无人机则插回卡车
            route = greedy_insert(data, route, [cid])
            route_set = set(route)

    return route, b2d

# ================== 迟到修复：局部重排算子（卡车路径重插入） ==================
def _late_repair_get_late_truck_customers(data, route, start_time, truck_speed, eps=1e-9):
    """返回卡车路径中“已迟到”的客户列表（按 late 从大到小）：[(late, cid), ...]"""
    arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
    late_list = []
    for cid in route:
        node = data.nodes[cid]
        if node.get('node_type') != 'customer':
            continue
        due = node.get('due_time', None)
        if due is None:
            continue
        svc = arrival_times.get(cid, None)
        if svc is None:
            continue
        late = svc - float(due)
        if late > eps:
            late_list.append((late, cid))
    late_list.sort(reverse=True, key=lambda x: x[0])
    return late_list

def _late_repair_best_reinsert_position(data, route, base_to_drone_customers, cid, ctx):
    """中文注释：卡车客户的“重插入”搜索（late-repair 的核心局部动作）。

    目的：在“不改变其他点集合”的前提下，通过把某个卡车服务客户 cid 从当前卡车路径中拿出来，
    再尝试插入到其它位置，寻找“总迟到更小（优先）/ 目标值更小（次优先）”的路径。

    返回：dict 或 None
      - None：cid 不在路径中，或路径太短，无法处理
      - dict：包含 old_pos/best_pos/base_cost/best_cost/base_late/best_late/best_route

    注意：这里的比较主次是 **先看总迟到 total_late**，再看目标值 cost。
    """
    if (route is None) or (len(route) < 3) or (cid not in route):
        return None

    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', sim.TRUCK_SPEED_UNITS))
    drone_speed = float(ctx.get('drone_speed', sim.DRONE_SPEED_UNITS))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 50.0))
    arrival_prefix = ctx.get('arrival_prefix', None)
    eps = float(ctx.get('eps', 1e-9))

    # 基准
    base_cost, _, _, _, _, base_late, _ = sim.evaluate_truck_drone_with_time(
        data,
        route,
        base_to_drone_customers,
        start_time,
        truck_speed,
        drone_speed,
        alpha_drone,
        lambda_late,
        arrival_prefix=arrival_prefix,
    )

    old_pos = int(route.index(cid))
    base_route = route[:]
    base_route.pop(old_pos)
    if len(base_route) < 2:
        return None

    # 默认：不改变（best 就是原路径）
    best_cost = float(base_cost)
    best_late = float(base_late)
    best_pos = int(old_pos)
    best_route = route

    # 端点一般固定：不允许插到第0位（起点），不允许插到最后一位之后（保持终点）
    # 插入位置 pos 表示插到 base_route[pos] 之前，因此 pos ∈ [1, len(base_route)-1]
    for pos in range(1, len(base_route)):
        trial = base_route[:]
        trial.insert(pos, cid)
        cost, _, _, _, _, total_late, _ = sim.evaluate_truck_drone_with_time(
            data,
            trial,
            base_to_drone_customers,
            start_time,
            truck_speed,
            drone_speed,
            alpha_drone,
            lambda_late,
            arrival_prefix=arrival_prefix,
        )

        # 先看迟到，再看目标
        if (float(total_late) + eps) < best_late or (
            abs(float(total_late) - best_late) <= eps and (float(cost) + eps) < best_cost
        ):
            best_cost = float(cost)
            best_late = float(total_late)
            best_pos = int(pos)
            best_route = trial

    return {
        'old_pos': old_pos,
        'best_pos': best_pos,
        'base_cost': float(base_cost),
        'best_cost': float(best_cost),
        'base_late': float(base_late),
        'best_late': float(best_late),
        'best_route': best_route,
    }



def _late_repair_score_bases_by_drone_lateness(
    data: 'Data',
    route: list,
    base_to_drone_customers: dict,
    start_time: float,
    truck_speed: float,
    drone_speed: float,
    # 中文注释：arrival_prefix 是“可选”的前缀到达时间缓存（用于加速），不传则为 None。
    # 注意：你当前环境是 Python 3.8，不支持 `dict | None` 这种 3.10+ 的类型写法。
    arrival_prefix=None,
    eps: float = 1e-9,
):
    """中文注释：粗略评估“每个基站导致的无人机客户迟到程度”。

    说明：这里的“服务时间”采用与 evaluate_truck_drone_with_time 一致的近似：
        t_service(c) = t_arrive(base) + 2 * dist(base,c) / drone_speed
    该近似不显式模拟同一基站多架无人机的排队/架次，仅用于指导“把哪个基站提前”这一局部动作。

    返回：列表[(sum_late, max_late, base_idx)]，按 sum_late 降序排列。
    """
    if not base_to_drone_customers:
        return []

    arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
    if arrival_prefix:
        # 中文注释：prefix 内固定的到达时间优先（用于动态场景：已走过的前缀不再改变）
        arrival_times = dict(arrival_times)
        arrival_times.update(arrival_prefix)

    scored = []
    for b, cs in base_to_drone_customers.items():
        b = int(b)
        if b not in arrival_times:
            # 中文注释：该基站不在当前 route（且不在 prefix），无法通过“提前基站”来修复
            continue
        t_b = float(arrival_times[b])
        sum_late = 0.0
        max_late = 0.0
        for c in cs:
            c = int(c)
            due = float(data.nodes[c].get('due_time', float('inf')))
            if not (due < float('inf')):
                continue
            d = float(data.costMatrix[b][c])
            t_svc = t_b + 2.0 * d / float(drone_speed)
            late = max(0.0, t_svc - due)
            if late > eps:
                sum_late += late
                if late > max_late:
                    max_late = late
        if sum_late > eps:
            scored.append((sum_late, max_late, b))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def _late_repair_best_base_reinsert(
    data: 'Data',
    route: list,
    base_to_drone_customers: dict,
    base_idx: int,
    ctx: dict,
):
    """中文注释：尝试把某个基站在卡车路径中“提前”，扫描所有插入位置找最优。"""
    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', 1.0))
    drone_speed = float(ctx.get('drone_speed', 1.0))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 0.0))
    arrival_prefix = ctx.get('arrival_prefix', None)
    eps = float(ctx.get('eps', 1e-9))

    try:
        base_cost, _, _, _, _, base_late, _ = sim.evaluate_truck_drone_with_time(
            data, route, base_to_drone_customers,
            start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late,
            arrival_prefix=arrival_prefix,
        )
    except Exception:
        return None

    if base_idx not in route:
        return None
    old_pos = route.index(base_idx)
    if old_pos <= 0 or old_pos >= len(route) - 1:
        return None

    route_wo = route[:old_pos] + route[old_pos + 1:]

    best = None
    best_cost = base_cost
    best_late = base_late
    best_pos = old_pos

    # 中文注释：保持起点(route[0])不动；也不把基站插到最后一个 depot 之后
    for pos in range(1, len(route_wo)):
        if pos == old_pos:
            # 中文注释：等价于不移动（因为已删除一个元素，pos==old_pos 不再是原位）
            pass
        trial = route_wo[:pos] + [base_idx] + route_wo[pos:]
        try:
            cost, _, _, _, _, total_late, _ = sim.evaluate_truck_drone_with_time(
                data, trial, base_to_drone_customers,
                start_time, truck_speed, drone_speed,
                alpha_drone, lambda_late,
                arrival_prefix=arrival_prefix,
            )
        except Exception:
            continue

        # 中文注释：优先降低迟到；若迟到相同，则看成本
        if (total_late + eps) < best_late or (abs(total_late - best_late) <= eps and cost + eps < best_cost):
            best_cost = cost
            best_late = total_late
            best_pos = pos
            best = trial

    if best is None:
        return None

    return {
        'old_pos': old_pos,
        'best_pos': best_pos,
        'base_cost': base_cost,
        'best_cost': best_cost,
        'base_late': base_late,
        'best_late': best_late,
        'best_route': best,
    }
def late_repair_truck_reinsert(data: Data, route: list, base_to_drone_customers: dict, ctx: dict):
    """中文注释：迟到修复（局部重排）。

    目标：在不做大规模破坏的前提下，通过少量“重插入”动作尽量降低迟到：
    1) 若主要迟到来自卡车客户：把最迟到的卡车客户从路径中取出，扫描所有插入位置找最小迟到/成本。
    2) 若主要迟到来自无人机客户：把造成迟到最多的“基站访问点”在卡车路径中提前（重插入）。

    说明：该算子是一个“修复型”局部搜索，默认只做少量步（max_moves）。
    """
    max_moves = int(ctx.get('LATE_REPAIR_MAX_MOVES', 3))
    dbg = bool(ctx.get('LATE_REPAIR_DEBUG', False))
    eps = float(ctx.get('eps', 1e-9))

    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', 1.0))
    drone_speed = float(ctx.get('drone_speed', 1.0))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 0.0))
    arrival_prefix = ctx.get('arrival_prefix', None)

    def _eval(r):
        return sim.evaluate_truck_drone_with_time(
            data, r, base_to_drone_customers,
            start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late,
            arrival_prefix=arrival_prefix,
        )

    # 初始评估
    try:
        best_cost, _, _, _, _, best_late, _ = _eval(route)
    except Exception:
        return route

    for _ in range(max_moves):
        if best_late <= eps:
            break

        # ---- 1) 找最迟到的卡车客户（只看当前 route 内的 customer 节点） ----
        arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
        if arrival_prefix:
            arrival_times = dict(arrival_times)
            arrival_times.update(arrival_prefix)

        worst_truck = None  # (late, cust_idx)
        for idx in route:
            nt = str(data.nodes[idx].get('node_type', '')).lower()
            if nt not in ('customer', 'truck_customer', 'truck'):
                continue
            due = float(data.nodes[idx].get('due_time', float('inf')))
            if not (due < float('inf')):
                continue
            t = float(arrival_times.get(idx, 0.0))
            late = max(0.0, t - due)
            if late > eps and (worst_truck is None or late > worst_truck[0]):
                worst_truck = (late, int(idx))

        # ---- 2) 找“导致无人机客户迟到最多”的基站 ----
        base_scores = _late_repair_score_bases_by_drone_lateness(
            data, route, base_to_drone_customers,
            start_time=start_time,
            truck_speed=truck_speed,
            drone_speed=drone_speed,
            arrival_prefix=arrival_prefix,
            eps=eps,
        )
        worst_base = base_scores[0] if base_scores else None  # (sum_late, max_late, base_idx)

        # 决策：优先修更“严重”的那一侧
        truck_signal = worst_truck[0] if worst_truck else 0.0
        drone_signal = worst_base[0] if worst_base else 0.0

        improved = False

        if truck_signal >= drone_signal and worst_truck is not None:
            # === 卡车客户重插入 ===
            cust = worst_truck[1]
            res = _late_repair_best_reinsert_position(data, route, base_to_drone_customers, cust, ctx)
            if res is not None:
                if dbg:
                    print(f"[REINS-DBG] cid={cust} old_pos={res['old_pos']} base_cost={res['base_cost']:.3f} base_late={res['base_late']:.3f}")
                    print(f"[REINS-DBG] best_pos={res['best_pos']} best_cost={res['best_cost']:.3f} best_late={res['best_late']:.3f} Δcost={res['best_cost']-res['base_cost']:.3f}")

                if (res['best_late'] + eps) < best_late or (abs(res['best_late'] - best_late) <= eps and res['best_cost'] + eps < best_cost):
                    route = res['best_route']
                    best_cost = res['best_cost']
                    best_late = res['best_late']
                    improved = True

        if (not improved) and worst_base is not None:
            # === 基站提前（降低无人机客户迟到） ===
            b = int(worst_base[2])
            resb = _late_repair_best_base_reinsert(data, route, base_to_drone_customers, b, ctx)
            if resb is not None:
                if dbg:
                    print(f"[BASE-INS-DBG] base={b} old_pos={resb['old_pos']} base_cost={resb['base_cost']:.3f} base_late={resb['base_late']:.3f}")
                    print(f"[BASE-INS-DBG] best_pos={resb['best_pos']} best_cost={resb['best_cost']:.3f} best_late={resb['best_late']:.3f} Δcost={resb['best_cost']-resb['base_cost']:.3f}")

                if (resb['best_late'] + eps) < best_late or (abs(resb['best_late'] - best_late) <= eps and resb['best_cost'] + eps < best_cost):
                    route = resb['best_route']
                    best_cost = resb['best_cost']
                    best_late = resb['best_late']
                    improved = True

        if not improved:
            # 中文注释：本轮没有找到可改进动作，提前结束
            break

    return route
def R_late_repair_reinsert(data, route, base_to_drone_customers, removed_customers, ctx):
    """迟到修复算子（Repair）：先完成常规修复（regret + drone），再做“迟到客户重插入”局部优化。"""
    res = R_regret_then_drone(data, route, base_to_drone_customers, removed_customers, ctx)
    # 中文注释：计数与调试（默认不输出；可在 ctx 里设置 LATE_REPAIR_DEBUG=True）
    ctx['_late_repair_calls'] = int(ctx.get('_late_repair_calls', 0)) + 1
    dbg = bool(ctx.get('LATE_REPAIR_DEBUG', False))
    if res is None:
        return None
    route2, b2d2 = res
    route3 = late_repair_truck_reinsert(data, route2, b2d2, ctx)
    if dbg:
        changed = (route3 != route2)
        # 只打印前 10 次，避免刷屏
        if ctx['_late_repair_calls'] <= 10:
            print(f"[LATE-REPAIR] call={ctx['_late_repair_calls']} changed={changed} len_route={len(route3)}")
        elif ctx['_late_repair_calls'] == 11:
            print("[LATE-REPAIR] ... (more calls omitted)")
    return route3, b2d2


def alns_truck_drone(data, base_to_drone_customers, max_iter=200, remove_fraction=0.1, T_start=1.0,
                     T_end=0.01, alpha_drone=0.3, lambda_late=50.0, truck_customers=None, use_rl=False,
                     rl_tau=0.5, rl_eta=0.1, start_idx=None, start_time: float = 0.0, bases_to_visit=None,
                     ctx=None):
    if ctx is None:
        ctx = {}

    # 1) drone_range：自动兼容你工程里的常量名
    if "drone_range" not in ctx or ctx["drone_range"] is None:
        ctx["drone_range"] = (
                globals().get("DRONE_RANGE_UNITS", None)
                or globals().get("DRONE_RANGE", None)
        )
    if ctx["drone_range"] is None:
        raise RuntimeError("[GUARD] 未找到无人机航程常量：请检查是否存在 DRONE_RANGE_UNITS 或 DRONE_RANGE")

    # 2) bases_to_visit：若没传，就默认所有基站（含 central）
    if not bases_to_visit:
        all_bases = [i for i, n in enumerate(data.nodes) if n.get("node_type") == "base"]
        if data.central_idx not in all_bases:
            all_bases.append(data.central_idx)
        ctx["bases_to_visit"] = all_bases
    else:
        ctx["bases_to_visit"] = list(bases_to_visit)

    ctx["alpha_drone"] = alpha_drone
    ctx["lambda_late"] = lambda_late
    ctx["truck_speed"] = sim.TRUCK_SPEED_UNITS
    ctx["drone_speed"] = sim.DRONE_SPEED_UNITS
    ctx["start_time"] = start_time

    # force_truck 集合（保证和 data 一致）
    ctx["force_truck_set"] = {i for i in data.customer_indices if data.nodes[i].get("force_truck", 0) == 1}

    if truck_customers is None:
        truck_customers = []

    # ---------- 必须覆盖客户集合（防止 repair 漏点导致 uncovered） ----------
    must_cover_set = set()
    must_cover_set |= set(truck_customers)
    for _b, _lst in base_to_drone_customers.items():
        try:
            must_cover_set |= set(_lst)
        except Exception:
            pass
    must_cover_set |= set(ctx.get("force_truck_set", set()))
    ctx["must_cover_set"] = set(must_cover_set)


    # 1) 初始卡车路径
    if start_idx is None:
        start_idx_used = data.central_idx
    else:
        start_idx_used = start_idx

    # protected 节点：任何 destroy/repair 都不得移除（包含：起点/终点/未来基站/force_truck）
    forced_customers = set(
        i for i, n in enumerate(data.nodes)
        if n.get('node_type') == 'customer' and int(n.get('force_truck', 0)) == 1
    )
    protected_nodes = set(ctx["bases_to_visit"]) | {start_idx_used, data.central_idx} | forced_customers

    current_route = nearest_neighbor_route_truck_only(
        data, truck_customers, start_idx=start_idx_used, end_idx=data.central_idx,
        bases_to_visit=ctx["bases_to_visit"]
    )

    dprint("[debug] init_route NODE_ID:", [data.nodes[i]['node_id'] for i in current_route])

    # 2) 当前/最优的 base_to_drone 也要拷贝
    current_b2d = {b: lst[:] for b, lst in base_to_drone_customers.items()}

    (current_cost,
     current_truck_dist,
     current_drone_dist,
     current_truck_late,
     current_drone_late,
     current_total_late,
     current_truck_time) = sim.evaluate_truck_drone_with_time(
        data,
        current_route,
        current_b2d,
        alpha_drone=alpha_drone,
        lambda_late=lambda_late,
        truck_speed=sim.TRUCK_SPEED_UNITS,
        drone_speed=sim.DRONE_SPEED_UNITS,
        start_time=start_time,
        arrival_prefix=ctx.get("arrival_prefix")
    )

    best_route = current_route[:]
    best_b2d = {b: lst[:] for b, lst in current_b2d.items()}
    best_cost = current_cost
    best_truck_dist = current_truck_dist
    best_drone_dist = current_drone_dist
    best_total_late = current_total_late
    best_truck_time = current_truck_time

    if ctx.get("verbose", False):
        print(f"初始: 成本={current_cost:.3f}, 卡车距={current_truck_dist:.3f}, 无人机距={current_drone_dist:.3f}, 总迟到={current_total_late:.3f}")

    # === RL: 每个算子的得分（Q 值） ===

    n_inner = len(current_route) - 2
    if n_inner <= 0:
        # 兜底：仍需保证 force_truck 约束一致
        best_route, best_b2d = enforce_force_truck_solution(data, best_route, best_b2d)
        return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time
    # ---------- ctx 处理：保留调用方传入的 dict 引用，用于回传算子统计 ----------
    ctx_in = ctx if isinstance(ctx, dict) else {}
    ctx = ctx_in
    ctx = build_ab_cfg(ctx)
    DESTROYS = ctx.get("DESTROYS", [D_random_route, D_worst_route, D_reloc_focus_v2, D_switch_coverage])
    REPAIRS = ctx.get("REPAIRS", [R_greedy_only, R_regret_only, R_greedy_then_drone, R_regret_then_drone,
                               R_late_repair_reinsert, R_base_feasible_drone_first])

    PAIRING_MODE = ctx.get("PAIRING_MODE", "free")  # "free" 或 "paired"
    ALLOWED_PAIRS = ctx.get("ALLOWED_PAIRS", None)  # paired 模式下用
    if ctx.get("verbose", False):
        print("[CFG-check]",
              "PAIRING_MODE=", PAIRING_MODE,
              "has_pairs=", bool(ALLOWED_PAIRS),
              "n_destroy=", len(DESTROYS),
              "n_repair=", len(REPAIRS),
              "scene_idx=", ctx.get("scene_idx", None))
    hit_stat = {}  # key: destroy_name -> dict(removed=0, hit=0, calls=0)
    op_stat = {}  # destroy -> dict(calls, accepts, best_hits, best_gain)

    for it in range(1, max_iter + 1):
        alpha = it / max_iter
        T = T_start * (1 - alpha) + T_end * alpha

        # --- 记录旧的 current_cost / best_cost，用来算奖励 ---
        old_current_cost = current_cost
        old_best_cost = best_cost

        # 1) 选择 destroy / repair（先都用 random，后面再加权/加RL）
        if PAIRING_MODE == "paired" and ALLOWED_PAIRS:
            D, R = random.choice(ALLOWED_PAIRS)
        else:
            D = random.choice(DESTROYS)
            R = random.choice(REPAIRS)

        dname = getattr(D, "__name__", "D?")
        rname = getattr(R, "__name__", "R?")
        # 统计建议按 (destroy, repair) 粒度记录，否则看不出“谁和谁搭配导致 late_fail/accept 变化”
        op_key = f"{dname}::{rname}"
        s = op_stat.setdefault(op_key, {})
        # 兜底补齐（避免 KeyError）
        s.setdefault("calls", 0)
        s.setdefault("accepts", 0)
        s.setdefault("best_hits", 0)
        s.setdefault("best_gain", 0.0)
        s.setdefault("late_fail", 0)
        s.setdefault("late_excess", 0.0)   # late_hard 被超出的总量（用于定位“多严重”）
        s.setdefault("repair_fail", 0)     # repair 返回 None 的次数（用于定位“可行性问题”）
        s.setdefault("cover_fail", 0)     # 漏点导致覆盖性失败次数
        s.setdefault("sa_reject", 0)
        s["calls"] += 1

        # 2) 生成候选解
        iter_ctx = dict(ctx)
        min_remove = int(iter_ctx.get("min_remove", 2))  # 默认 2，动态建议传 3
        iter_ctx["num_remove"] = max(min_remove, int(remove_fraction * n_inner))
        iter_ctx["protected_nodes"] = protected_nodes

        # 后面你做定制算子时，在 ctx 里加 C_moved_accept / C_boundary / bases_to_visit 等

        destroyed_route, destroyed_b2d, removed_customers = D(data, current_route, current_b2d, iter_ctx)
        # === B1: destroy 命中率统计（只在 verbose=True 时打印，套件不刷屏）===
        if iter_ctx.get("verbose", False):
            removed_set = set(removed_customers or [])
            if len(removed_set) > 0:
                C1 = set(iter_ctx.get("C_moved_accept", set()))
                C2 = set(iter_ctx.get("C_force_truck", set()))
                C3 = set(iter_ctx.get("C_boundary", set()))
                hit = len(removed_set & (C1 | C2 | C3))

                name = getattr(D, "__name__", "D?")
                hs = hit_stat.setdefault(name, {"removed": 0, "hit": 0, "calls": 0})
                hs["removed"] += len(removed_set)
                hs["hit"] += hit
                hs["calls"] += 1

                # if it % 100 == 0:
                #     print("[HIT-SUMMARY] it=", it)
                #     for k, v in sorted(hit_stat.items()):
                #         rate = v["hit"] / max(1, v["removed"])
                #         print(
                #             f"  {k:18s} calls={v['calls']:4d} removed={v['removed']:4d} hit={v['hit']:4d} rate={rate:.2f}")

        cand_route, cand_b2d = R(data, destroyed_route, destroyed_b2d, removed_customers, iter_ctx)

        if cand_route is None or cand_b2d is None:
            s["repair_fail"] += 1
            continue

        # 强制约束修复（最终兜底）
        cand_route, cand_b2d = enforce_force_truck_solution(data, cand_route, cand_b2d)


        # ---------- 覆盖性护栏：候选解不得漏掉 must_cover_set 里的客户 ----------
        must_cover = set(iter_ctx.get("must_cover_set", set()))
        if must_cover:
            route_cust = {i for i in cand_route if 0 <= i < len(data.nodes) and data.nodes[i].get("node_type") == "customer"}
            drone_cust = {int(c) for _b, _lst in cand_b2d.items() for c in _lst}
            missing_cov = must_cover - (route_cust | drone_cust)
            if missing_cov:
                s.setdefault("cover_fail", 0)
                s["cover_fail"] += 1
                continue

        # === 3) 评估候选解 ===
        (cand_cost,
         cand_truck_dist,
         cand_drone_dist,
         cand_truck_late,
         cand_drone_late,
         cand_total_late,
         cand_truck_time) = sim.evaluate_truck_drone_with_time(
            data,
            cand_route,
            cand_b2d,
            alpha_drone=alpha_drone,
            lambda_late=lambda_late,
            truck_speed=sim.TRUCK_SPEED_UNITS,
            drone_speed=sim.DRONE_SPEED_UNITS,
            start_time=start_time,
            arrival_prefix=ctx.get("arrival_prefix")
        )

        late_abs = float(iter_ctx.get("late_hard", float("inf")))
        late_delta = iter_ctx.get("late_hard_delta", None)
        # 1) 迟到“增量”硬护栏：避免因总迟到阈值过小导致候选几乎全被丢弃
        if late_delta is not None:
            late_delta = float(late_delta)
            inc = float(cand_total_late - current_total_late)
            if inc > late_delta + 1e-9:
                s.setdefault("late_delta_fail", 0)
                s.setdefault("late_delta_excess", 0.0)
                s["late_delta_fail"] += 1
                s["late_delta_excess"] += float(inc - late_delta)
                continue
        # 2) 迟到“绝对值”硬护栏：兜底避免迟到爆炸
        if cand_total_late > late_abs + 1e-9:
            s["late_fail"] += 1
            s["late_excess"] += float(cand_total_late - late_abs)
            continue

        # === 4) SA 接受准则 ===
        delta = cand_cost - current_cost
        if delta < 0:
            accept = True
        else:
            prob = math.exp(-delta / max(T, 1e-6))
            accept = (random.random() < prob)

        if accept:
            current_route = cand_route
            current_b2d = {b: lst[:] for b, lst in cand_b2d.items()}
            current_cost = cand_cost
            current_truck_dist = cand_truck_dist
            current_drone_dist = cand_drone_dist
            current_total_late = cand_total_late
            current_truck_time = cand_truck_time
            s["accepts"] += 1

            if cand_cost < best_cost:
                gain = best_cost - cand_cost  # 先用旧 best_cost 计算收益（>0）
                best_route = current_route[:]
                best_b2d = {b: lst[:] for b, lst in current_b2d.items()}
                best_cost = cand_cost
                best_truck_dist = cand_truck_dist
                best_drone_dist = cand_drone_dist
                best_total_late = cand_total_late
                best_truck_time = cand_truck_time

                s["best_hits"] += 1
                s["best_gain"] += gain
        else:
            # 仅统计：SA 没接受（不改变任何决策逻辑）
            s["sa_reject"] += 1
        # if ctx.get("verbose", False) and it % 200 == 0:
        #     print("[OP-SUMMARY] it=", it)
        #     for k, v in sorted(op_stat.items()):
        #         calls = v.get("calls", 0)
        #         acc = v.get("accepts", 0)
        #         latef = v.get("late_fail", 0)
        #         sarej = v.get("sa_reject", 0)
        #         besth = v.get("best_hits", 0)
        #         bestg = v.get("best_gain", 0.0)
        #         repf = v.get("repair_fail", 0)
        #         lex = v.get("late_excess", 0.0)
        #
        #         acc_rate = acc / max(1, calls)
        #         late_rate = latef / max(1, calls)
        #         rep_rate = repf / max(1, calls)
        #         sa_rate = sarej / max(1, calls)
        #         best_rate = besth / max(1, calls)
        #         avg_gain = bestg / max(1, besth)
        #         avg_late_excess = lex / max(1, latef)
        #
        #         print(f"  {k:18s} calls={calls:4d} acc={acc:4d} acc_rate={acc_rate:.2f} "
        #               f"repair_fail={repf:4d} rep_rate={rep_rate:.2f} "
        #               f"late_fail={latef:4d} late_rate={late_rate:.2f} avg_late_excess={avg_late_excess:.3f} "
        #               f"sa_rej={sarej:4d} sa_rate={sa_rate:.2f} "
        #               f"best_hits={besth:4d} best_rate={best_rate:.2f} "
        #               f"avg_best_gain={avg_gain:.3f} total_best_gain={bestg:.3f}")
    # 打印偏爱算子
    dprint("[debug] best_route NODE_ID:", [data.nodes[i]['node_id'] for i in best_route])

    # 返回前再做一次兜底约束（避免 force_truck 客户被算子转走导致后续场景丢失）
    best_route, best_b2d = enforce_force_truck_solution(data, best_route, best_b2d)

    # ---------- 回传算子统计：用于消融验证（不影响求解逻辑） ----------
    try:
        ctx_in["__op_stat"] = op_stat
        ctx_in["__hit_stat"] = hit_stat
    except Exception:
        pass

    # if bool(ctx.get("verbose_stat", True)):
    #     _print_operator_stats(op_stat, top_k=int(ctx.get("verbose_stat_topk", 30)))
    return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time

def get_drone_served_before_t(full_b2d_cur, full_finish_cur, t, eps=1e-9):
    # t时刻之前已服务的无人机客户
    served = set()
    for b, cs in full_b2d_cur.items():
        for c in cs:
            if full_finish_cur.get(c, float('inf')) <= t + eps:
                served.add(c)
    return sorted(served)

def quick_filter_relocations(
    data_cur,
    data_prelim,
    full_route_cur,
    full_b2d_cur,
    req_clients,
    decisions,
    alpha_drone,
    lambda_late,
    truck_speed,
    drone_speed,
    delta_cost_max=30.0,
    delta_late_max=0.10,
    prefix_route=None,
):
    """    中文注释：对 apply_relocations_for_decision_time 的初判 decisions 做二次筛选（quick_filter）。

    在原 quick_filter 的基础上，按“锁死规则”加入：
    - 冻结承诺窗 PROM_READY/PROM_DUE；
    - 事件接受后落地 EFFECTIVE_DUE（本轮事件 L）；拒绝则回退到 PROM_DUE；
    - 输出每条请求的 Δcost、Δlate_prom（按 PROM_DUE）、Δlate_eff（按 EFFECTIVE_DUE）。

    返回：data_next, decisions_filtered, qf_deltas
      - qf_deltas[cid] = {"D_COST","D_LATE_PROM","D_LATE_EFF"}
    """
    debug = bool(globals().get("DEBUG_QUICK_FILTER", False))

    # ---------- 小工具：在 b2d 里找客户归属基站 ----------
    def _find_base_of_client(b2d, cid):
        for b, lst in b2d.items():
            if cid in lst:
                return b
        return None

    def _remove_from_b2d(b2d, cid):
        for b in list(b2d.keys()):
            if cid in b2d[b]:
                b2d[b] = [x for x in b2d[b] if x != cid]

    def _move_to_base(b2d, cid, b_new):
        _remove_from_b2d(b2d, cid)
        if b_new is None:
            return
        if b_new not in b2d:
            b2d[b_new] = []
        if cid not in b2d[b_new]:
            b2d[b_new].append(cid)

    def _remove_from_route(route, cid):
        # route 里理论上不会重复，但这里写稳健一点
        return [x for x in route if x != cid]

    def _ensure_in_route_tail(route, cid, central_idx):
        # 最小做法：把 cid 插到“最后一个 depot(central)”前面，避免破坏前缀（保守但一致）
        if cid in route:
            return route
        if len(route) >= 1 and route[-1] == central_idx:
            return route[:-1] + [cid] + [central_idx]
        return route + [cid]

    def _cheapest_reinsert_for_truck(data, route, cid, min_pos, central_idx):
        """        中文注释：truck→truck 的局部最小代价动作（cheapest-insertion）。
        目的：避免仅按“原访问顺序”评估坐标变更时误判。

        在不改动冻结前缀（min_pos 之前部分）的前提下，把 cid 移除后扫描插入位置选最小增量。
        返回：new_route, old_pos, best_pos, best_delta
        """
        if route is None:
            return [], None, None, 0.0
        if (cid not in route) or (len(route) < 3):
            return list(route), None, None, 0.0

        r = list(route)
        try:
            old_pos = r.index(cid)
        except Exception:
            return r, None, None, 0.0

        # 不允许动冻结前缀：若 cid 在前缀内，直接不动
        if old_pos < min_pos:
            return list(route), old_pos, old_pos, 0.0

        # 移除
        r.pop(old_pos)

        # 允许插入到最后一个 central 之前
        try:
            if len(r) > 0 and r[-1] == central_idx:
                max_pos = len(r) - 1
            else:
                max_pos = len(r)
        except Exception:
            max_pos = len(r)

        best_pos = None
        best_delta = float("inf")

        def _dist(i, j):
            return float(data.costMatrix[int(i)][int(j)])

        for pos in range(int(min_pos), int(max_pos) + 1):
            if pos <= 0 or pos >= len(r):
                continue
            a = r[pos - 1]
            b = r[pos]
            delta = _dist(a, cid) + _dist(cid, b) - _dist(a, b)
            if delta < best_delta:
                best_delta = delta
                best_pos = pos

        if best_pos is None:
            return list(route), old_pos, old_pos, 0.0

        r.insert(best_pos, cid)
        return r, old_pos, best_pos, best_delta

    # ---------- 1) 统一 decisions 输入格式 ----------
    # apply_relocations_for_decision_time: (cid, nid, decision, reason, old_x, old_y, nx, ny)
    # quick_filter 内/外部使用：         (cid, decision, nx, ny, reason)
    decisions_norm = []
    for it in decisions:
        if isinstance(it, (list, tuple)) and len(it) == 8:
            cid, _nid, dec, reason, _ox, _oy, nx, ny = it
            decisions_norm.append((int(cid), str(dec), float(nx), float(ny), str(reason)))
        elif isinstance(it, (list, tuple)) and len(it) == 5:
            cid, dec, nx, ny, reason = it
            decisions_norm.append((int(cid), str(dec), float(nx), float(ny), str(reason)))
        else:
            raise ValueError(f"quick_filter_relocations: 无法识别 decisions 元素格式: {it}")

    # ---------- 2) baseline：从 data_cur 出发评估（不包含任何新变更） ----------
    data_work = copy.deepcopy(data_cur)
    route_work = list(full_route_cur) if full_route_cur is not None else []
    b2d_work = copy.deepcopy(full_b2d_cur) if full_b2d_cur is not None else {}

    # ---------- 2.1) 冻结前缀：cheapest-insertion 不允许把点插到已服务前缀之前 ----------
    prefix_end_pos = 0
    try:
        if prefix_route is not None and isinstance(prefix_route, (list, tuple)) and len(prefix_route) > 0:
            pr = list(prefix_route)
            if route_work[:len(pr)] == pr:
                prefix_end_pos = len(pr) - 1
            else:
                if pr[-1] in route_work:
                    prefix_end_pos = route_work.index(pr[-1])
    except Exception:
        prefix_end_pos = 0
    min_insert_pos = max(1, int(prefix_end_pos) + 1)

    recompute_cost_and_nearest_base(data_work)
    res_work = sim.evaluate_full_system(
        data_work, route_work, b2d_work,
        alpha_drone, lambda_late,
        truck_speed, drone_speed
    )
    # 中文注释：按 PROM_DUE 口径的 baseline 迟到（用于 Δlate_prom）
    late_prom_work = _total_late_against_due(
        data_work, route_work, b2d_work, res_work,
        due_mode="prom", drone_speed=drone_speed
    )

    qf_deltas = {}
    decisions_filtered = []

    # ---------- 3) 逐条叠加尝试 ACCEPT ----------
    for (cid, decision, nx, ny, reason) in decisions_norm:
        # 统一拿到 PROM_DUE
        try:
            prom_due = float(data_work.nodes[cid].get("prom_due", data_work.nodes[cid].get("due_time", float("inf"))))
        except Exception:
            prom_due = float("inf")

        # 若初判就不是 ACCEPT：按锁死规则回退 EFFECTIVE_DUE= PROM_DUE
        if decision != "ACCEPT":
            try:
                data_work.nodes[cid]["effective_due"] = prom_due
                data_work.nodes[cid]["due_time"] = prom_due
                # baseline 更新（后续请求以回退后的 due 继续评估）
                res_work = sim.evaluate_full_system(
                    data_work, route_work, b2d_work,
                    alpha_drone, lambda_late,
                    truck_speed, drone_speed
                )
                late_prom_work = _total_late_against_due(
                    data_work, route_work, b2d_work, res_work,
                    due_mode="prom", drone_speed=drone_speed
                )
            except Exception:
                pass
            qf_deltas[int(cid)] = {"D_COST": 0.0, "D_LATE_PROM": 0.0, "D_LATE_EFF": 0.0}
            decisions_filtered.append((cid, "REJECT", nx, ny, reason))
            continue

        # ---------- accept trial ----------
        node_pre = data_prelim.nodes[cid]
        upd_force = int(node_pre.get("force_truck", 0))
        upd_base_lock = node_pre.get("base_lock", None)
        try:
            upd_base_lock = int(upd_base_lock) if upd_base_lock is not None else None
        except Exception:
            upd_base_lock = None

        # 中文注释：candidate_effective_due 是 apply 阶段根据 DELTA_AVAIL_H 计算的 L
        cand_due = node_pre.get("candidate_effective_due", None)
        if cand_due is None:
            # 兜底：若 apply 阶段未写入，则在此计算
            try:
                pr = float(node_pre.get("prom_ready", node_pre.get("ready_time", 0.0)))
                pd = float(node_pre.get("prom_due", node_pre.get("due_time", float("inf"))))
                d_av = float(node_pre.get("delta_avail_h", 0.0))
                cand_due = max(pd, pr + d_av)
            except Exception:
                cand_due = None
        try:
            cand_due = float(cand_due) if cand_due is not None else None
        except Exception:
            cand_due = None

        upd = {
            "x": float(nx),
            "y": float(ny),
            "force_truck": upd_force,
            "base_lock": upd_base_lock,
            "coord_source": node_pre.get("coord_source", "PERTURBED"),
        }

        # trial：复制当前 work 状态
        data_trial = copy.deepcopy(data_work)
        route_trial = list(route_work)
        b2d_trial = copy.deepcopy(b2d_work)

        # 1) 更新坐标/标记
        node_t = data_trial.nodes[cid]
        node_t["x"] = upd["x"]
        node_t["y"] = upd["y"]
        node_t["force_truck"] = upd["force_truck"]
        node_t["base_lock"] = upd["base_lock"]
        node_t["coord_source"] = upd["coord_source"]

        # 2) 落地有效截止时间（EFFECTIVE_DUE）
        if cand_due is not None:
            node_t["effective_due"] = cand_due
            node_t["due_time"] = cand_due

        # 3) 同步更新评估输入解（route/b2d）
        central_idx = getattr(data_trial, "central_idx", 0)
        old_base = _find_base_of_client(b2d_trial, cid)
        old_in_route = (cid in route_trial)

        if upd["force_truck"] == 1:
            _remove_from_b2d(b2d_trial, cid)
            route_trial = _ensure_in_route_tail(route_trial, cid, central_idx)
        else:
            if upd["base_lock"] is not None:
                route_trial = _remove_from_route(route_trial, cid)
                _move_to_base(b2d_trial, cid, upd["base_lock"])

        recompute_cost_and_nearest_base(data_trial)

        # 2.5) truck→truck 坐标变更：允许一次 cheapest-insertion
        if old_in_route and (upd["force_truck"] == 0) and (upd["base_lock"] is None):
            route_trial2, old_pos, best_pos, best_delta = _cheapest_reinsert_for_truck(
                data_trial, route_trial, cid, min_insert_pos, central_idx
            )
            route_trial = route_trial2
            if debug and (old_pos is not None) and (best_pos is not None) and (best_pos != old_pos):
                nid = data_cur.nodes[cid].get("node_id", cid)
                print(
                    f"[QF-INS] cid={cid} nid={nid} old_pos={old_pos} best_pos={best_pos} "
                    f"best_delta={best_delta:.3f} min_pos={min_insert_pos}"
                )

        # 4) 评估 trial
        res1 = sim.evaluate_full_system(
            data_trial, route_trial, b2d_trial,
            alpha_drone, lambda_late,
            truck_speed, drone_speed
        )
        late_prom_trial = _total_late_against_due(
            data_trial, route_trial, b2d_trial, res1,
            due_mode="prom", drone_speed=drone_speed
        )

        # Δ 指标
        d_truck = float(res1["truck_dist"]) - float(res_work["truck_dist"])
        d_drone = float(res1["drone_dist"]) - float(res_work["drone_dist"])
        d_late_eff = float(res1["total_late"]) - float(res_work["total_late"])
        d_cost = float(res1["cost"]) - float(res_work["cost"])
        d_late_prom = float(late_prom_trial) - float(late_prom_work)

        qf_deltas[int(cid)] = {
            "D_COST": float(d_cost),
            "D_LATE_PROM": float(d_late_prom),
            "D_LATE_EFF": float(d_late_eff),
        }

        if debug:
            nid = data_cur.nodes[cid].get("node_id", cid)
            new_base = _find_base_of_client(b2d_trial, cid)
            typ0 = f"drone@base{old_base}" if old_base is not None else "truck"
            typ1 = f"drone@base{new_base}" if new_base is not None else "truck"
            print(
                f"[QF] cid={cid} nid={nid} {typ0}->{typ1} "
                f"force={upd_force} base_lock={upd_base_lock} in_route0={old_in_route} "
                f"Δtruck={d_truck:.3f} Δdrone={d_drone:.3f} Δlate_eff={d_late_eff:.3f} "
                f"Δlate_prom={d_late_prom:.3f} Δcost={d_cost:.3f}"
            )

        # 阈值判定（保持原先 quick_filter 策略：cost 与 late_eff 双阈值）
        if (d_cost > float(delta_cost_max)) or (d_late_eff > float(delta_late_max)):
            # 按锁死规则回退 EFFECTIVE_DUE= PROM_DUE（即使这次被拒绝）
            try:
                data_work.nodes[cid]["effective_due"] = prom_due
                data_work.nodes[cid]["due_time"] = prom_due
                res_work = sim.evaluate_full_system(
                    data_work, route_work, b2d_work,
                    alpha_drone, lambda_late,
                    truck_speed, drone_speed
                )
                late_prom_work = _total_late_against_due(
                    data_work, route_work, b2d_work, res_work,
                    due_mode="prom", drone_speed=drone_speed
                )
            except Exception:
                pass
            decisions_filtered.append(
                (
                    cid, "REJECT", nx, ny,
                    "quick_filter拒绝（未落地坐标/模式变更；并回退有效窗）："
                    f"Δtruck={d_truck:.3f}, Δdrone={d_drone:.3f}, "
                    f"Δlate_eff={d_late_eff:.3f}, Δlate_prom={d_late_prom:.3f}, Δcost={d_cost:.3f}；"
                    f"候选动作：{reason}"
                )
            )
            continue

        # 通过：提交 trial -> work，并更新 baseline
        data_work = data_trial
        route_work = route_trial
        b2d_work = b2d_trial
        res_work = res1
        late_prom_work = late_prom_trial
        decisions_filtered.append((cid, "ACCEPT", nx, ny, reason))

    return data_work, decisions_filtered, qf_deltas
def _apply_xy(node, x, y, source="PERTURBED"):
    node["x"] = x
    node["y"] = y
    node["coord_source"] = source  # 关键：同步标记

def run_one(file_path: str, seed: int, ab_cfg: dict, perturbation_times=None, enable_plot: bool = False, verbose: bool = True, events_path: str = "", decision_log_path: str = ""):
    if verbose:
        print("[CFG-IN]", ab_cfg.get("PAIRING_MODE"), ab_cfg.get("late_hard"),
          len(ab_cfg.get("DESTROYS", [])), len(ab_cfg.get("REPAIRS", [])))

    if perturbation_times is None:
        perturbation_times = []
    # 统一过滤/去重/排序，避免传入 0 导致重复场景、以及不同运行方式输出不一致
    perturbation_times = _normalize_perturbation_times(perturbation_times)
    def cover_uncovered_by_truck_suffix(data, full_route, full_b2d, prefix_len, verbose=False):
        """
        动态拼接后工程兜底：若出现未覆盖客户（既不在卡车路径也不在无人机任务），
        则把它们强制插入到“后缀部分”的卡车路径中（不动前缀已执行部分），避免丢点崩溃。
        """
        all_customers = {i for i, n in enumerate(data.nodes) if n.get("node_type") == "customer"}
        truck_customers = {i for i in full_route
                           if 0 <= i < len(data.nodes) and data.nodes[i].get("node_type") == "customer"}
        drone_customers = {c for cs in full_b2d.values() for c in cs}
        uncovered = sorted(all_customers - (truck_customers | drone_customers))
        if not uncovered:
            return full_route

        if verbose:
            nids = [data.nodes[i].get("node_id", i) for i in uncovered]
            print(f"[GUARD][COVER] 发现未覆盖客户 {len(uncovered)} 个，强制插入卡车后缀路径 node_id={nids}")

        r = full_route[:]
        if len(r) < 2:
            r.extend(uncovered)
            return r

        start_pos = max(int(prefix_len), 1)
        end_pos = max(len(r) - 1, start_pos)

        def delta_cost(route, node_idx, pos):
            a = route[pos - 1]
            b = route[pos]
            return float(sim.truck_arc_cost(data, a, node_idx) + sim.truck_arc_cost(data, node_idx, b) - sim.truck_arc_cost(data, a, b))

        for c in uncovered:
            # 标记 force_truck，避免后续又被分给无人机
            try:
                data.nodes[c]["force_truck"] = 1
            except Exception:
                pass

            best_pos = end_pos
            best_delta = float("inf")
            for pos in range(start_pos, end_pos + 1):
                try:
                    d = delta_cost(r, c, pos)
                except Exception:
                    continue
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            r.insert(best_pos, c)

        return r

    if seed is not None:
        set_seed(int(seed))

    ab_cfg = build_ab_cfg(ab_cfg)
    # ===================== 1) 读取数据（场景0：全原始坐标）=====================
    data = read_data(file_path, scenario=0, strict_schema=True)
    if verbose:
        print_tw_stats(data)  # 或者 print_tw_stats(data_cur)
    # 可选：schema 对齐检查
    try:
        if hasattr(data, "schema_cols") and "CSV_REQUIRED_COLS" in globals():
            # 中文注释：允许 nodes.csv 额外列存在（例如 *_promise.csv 的 PROM_* 列），只要必需列不缺失即可。
            _cols = list(getattr(data, "schema_cols", []) or [])
            _missing = [c for c in CSV_REQUIRED_COLS if c not in _cols]
            if _missing:
                raise RuntimeError(f"数据 schema 缺失必需列: {_missing}；请检查 data_io 的 CSV_REQUIRED_COLS")
    except Exception as e:
        raise

    if verbose:
        print(f"节点数: {len(data.nodes)}, 中心仓库 idx: {data.central_idx}")

    # ===================== [OFFLINE-EVENTS] 读取 events.csv（若提供）=====================
    nodeid2idx = {int(n.get("node_id")): i for i, n in enumerate(data.nodes)}
    offline_events = []
    offline_groups = None
    decision_log_rows = []
    if events_path:
        try:
            offline_events = load_events_csv(events_path)
        except Exception as _e:
            raise RuntimeError(f"[OFFLINE] events.csv 读取失败：{events_path}，err={_e}")
        if not offline_events:
            raise RuntimeError(f"[OFFLINE] events_path 提供但读取为空：{events_path}")
        offline_groups = group_events_by_time(offline_events)
        if verbose:
            print(f"[OFFLINE] load events: {events_path}, events={len(offline_events)}")
        # 用 events.csv 中出现过的 EVENT_TIME 覆盖决策点集合（支持非连续/非整数）
        _ts = sorted({round(float(e.get('EVENT_TIME', 0.0)), 6) for e in offline_events})
        perturbation_times = [float(t) for t in _ts if float(t) > 0.0]
        if verbose:
            print(f"[OFFLINE] decision times overridden by events: T=1..{len(perturbation_times)}")
    # ===================== 2) 初始分类（场景0）=====================
    base_to_drone_customers, truck_customers = sim.classify_clients_for_drone(data)
    if verbose:
        print("需要卡车服务的客户数:", len(truck_customers))
        print("各基站无人机客户数:", {b: len(cs) for b, cs in base_to_drone_customers.items()})

    # [PROMISE] 场景0/动态阶段不再对无人机时间窗做 relax/reset（承诺窗由场景0 ETA0 冻结生成）
    if False:
            relax_drone_time_windows(
                data, base_to_drone_customers,
                extra_slack=2.0,
                reset_ready_to_zero=True
            )

    # ===================== 3) 场景0：跑一次 ALNS（No-RL）=====================
    if verbose:
        print("\n===== Advanced ALNS (No RL, official solution) =====")

    ctx0 = dict(ab_cfg)  # 关键：场景0也吃实验配置（paired/free/算子池）
    ctx0["verbose"] = verbose

    # [PROMISE] 场景0不计迟到：避免 late_hard 护栏误伤（即使你实验配置里开启了 late_hard）
    ctx0["late_hard"] = 1e18
    ctx0["late_hard_delta"] = 1e18

    (best_route,
     best_b2d,
     best_cost,
     best_truck_dist,
     best_drone_dist,
     best_total_late,
     best_truck_time) = alns_truck_drone(
        data,
        base_to_drone_customers,
        max_iter=1000,
        remove_fraction=0.1,
        T_start=1.0,
        T_end=0.01,
        alpha_drone=0.3,
        lambda_late=0.0,
        truck_customers=truck_customers,
        use_rl=False,
        rl_tau=0.5,
        rl_eta=0.1,
        ctx=ctx0
    )

    arrival_times, total_time, total_late = sim.compute_truck_schedule(
        data, best_route, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
    )
    depart_times, finish_times, base_finish_times = compute_multi_drone_schedule(
        data, best_b2d, arrival_times,
        num_drones_per_base=NUM_DRONES_PER_BASE,
        drone_speed=DRONE_SPEED_UNITS
    )

    # ===================== [PROMISE] 3.5) 用场景0 ETA0 生成并冻结平台承诺窗 =====================
    # 中文注释：场景0不考虑时间窗/迟到，仅用于生成“平台承诺窗口”（PROM_READY/PROM_DUE），并冻结用于后续所有场景。
    # 护栏：若输入本身已经是 *_promise.csv，则认为承诺窗已冻结，避免再次生成并输出 _promise_promise.csv。
    if _is_promise_nodes_file(file_path):
        freeze_existing_promise_windows_inplace(data)
        if verbose:
            print(f"[PROMISE] input already *_promise.csv, skip regenerate/write: {file_path}")
    else:
        _full_eval0_tmp = sim.evaluate_full_system(
            data, best_route, best_b2d,
            alpha_drone=0.3, lambda_late=0.0,
            truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=DRONE_SPEED_UNITS
        )
        eta0_map = compute_eta_map(data, best_route, best_b2d, _full_eval0_tmp, drone_speed=DRONE_SPEED_UNITS)
        apply_promise_windows_inplace(data, eta0_map, promise_width_h=0.5)

        # 输出 nodes_*_promise.csv（不覆盖原始数据集）
        try:
            promise_nodes_path = _derive_promise_nodes_path(file_path)
            write_promise_nodes_csv(file_path, promise_nodes_path, eta0_map, promise_width_h=0.5)
            if verbose:
                print(f"[PROMISE] wrote: {promise_nodes_path}")
        except Exception as _e:
            print(f"[PROMISE-WARN] 写出 promise nodes 失败: {_e}")

    # 统一口径：全系统完成时刻（卡车到达客户/基站 + 无人机完成）
    finish_all_times = dict(arrival_times)
    finish_all_times.update(finish_times)
    system_finish_time = max(total_time, max(base_finish_times.values()) if base_finish_times else 0.0)

    sim.check_disjoint(data, best_route, best_b2d)
    if verbose:
        print("最优卡车路径（按 NODE_ID）:", [data.nodes[i]['node_id'] for i in best_route])
        print(f"最终: 成本={best_cost:.3f}, 卡车距={best_truck_dist:.3f}, "
              f"无人机距={best_drone_dist:.3f}, 总迟到={best_total_late:.3f}, "
              f"卡车总时间={best_truck_time:.2f}h, 系统完成时间={system_finish_time:.2f}h")
        print("各基站完成时间：")
        for b, t_fin in base_finish_times.items():
            n = data.nodes[b]
            print(f"  base node_id={n['node_id']}, type={n['node_type']}, 完成时间={t_fin:.2f}h")

    if enable_plot:
        visualize_truck_drone(data, best_route, best_b2d, title="Scenario 0: original (no relocation)")

    # ===================== 4) 结果表：先记场景0（FULL口径）=====================
    scenario_results = []
    full_eval0 = sim.evaluate_full_system(
        data, best_route, best_b2d,
        alpha_drone=0.3, lambda_late=0.0,
        truck_speed=TRUCK_SPEED_UNITS, drone_speed=DRONE_SPEED_UNITS
    )

    # [PROMISE] 场景0：输出 late_prom/late_eff（late_eff 以冻结窗为准）
    _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
    emit_scene_late_logs(_late_dir, scene_idx=0, decision_time=0.0, data=data, full_route=best_route, full_b2d=best_b2d, full_eval=full_eval0, prefix="", drone_speed=DRONE_SPEED_UNITS)
    # 中文注释：scene=0（初始静态解）也输出迟到分解
    if DEBUG_LATE and ((DEBUG_LATE_SCENES is None) or (0 in DEBUG_LATE_SCENES)):
        # 中文注释：debug_print_lateness_topk 已在 slim 版本中移除（避免控制台大输出拖慢实验）。
        # 如需查看 TopK 迟到客户，请查 late_logs/*.csv（emit_scene_late_logs 会写出）。
        pass

    scenario_results.append(_pack_scene_record(0, 0.0, full_eval0, num_req=0, num_acc=0, num_rej=0, alpha_drone=0.3, lambda_late=50.0))
    global_xlim, global_ylim = compute_global_xlim_ylim(
        data=data,
        reloc_radius=ab_cfg.get("reloc_radius", 0.8),
        pad_min=5.0,
        step_align=10.0
    )

    # ===================== 5) 动态循环初始化（“全局完整口径”状态）=====================
    if perturbation_times:
        data_cur = data

        full_route_cur = best_route.copy()
        full_b2d_cur = {b: cs.copy() for b, cs in best_b2d.items()}

        full_arrival_cur = arrival_times  # 全局从0开始
        full_depart_cur = depart_times  # 全局从0开始
        full_finish_cur = finish_all_times  # 全局从0开始（包含卡车+无人机完成时刻）

        scene_idx = 1
        t_prev = 0.0  # 上一决策时刻（只用来取“请求窗口”）
        # ---------- 5.0 动态请求流：仅由 events.csv 驱动（不在求解阶段随机生成请求） ----------
        reloc_radius = float(ab_cfg.get("reloc_radius", 0.8)) if ab_cfg else 0.8
        if offline_groups is None:
            raise RuntimeError("动态模式需要提供 events_path（events.csv），以保证请求流可复现且公平")
        if verbose:
            print(f"[RELOC-PLAN] mode=events radius={reloc_radius}")


        decision_times_list = [float(x) for x in perturbation_times]
        for _k_dec, decision_time in enumerate(decision_times_list):

            # ---------- 决策时刻护栏：不得逆序 ----------
            if decision_time < t_prev - 1e-9:
                raise RuntimeError(f"[GUARD] 决策时刻逆序：t_prev={t_prev}, t_dec={decision_time}")

            print(f"\n===== 场景 {scene_idx}: 决策时刻 t = {decision_time:.2f} h 应用位置变更后 =====")
            if seed is not None:
                set_seed(int(seed) + int(scene_idx))

            # ---------- 5.1 split：在决策点前已服务/未服务 + 虚拟位置 ----------
            served_nodes, remaining_nodes, current_node, virtual_pos, prefix_route = split_route_by_decision_time(
                full_route_cur,
                full_arrival_cur,
                decision_time,
                data_cur.central_idx,
                data_cur
            )
            prefix_route_for_plot = prefix_route[:]

            served_nids = [data_cur.nodes[i]['node_id'] for i in served_nodes]
            remaining_nids = [data_cur.nodes[i]['node_id'] for i in remaining_nodes]
            if verbose:
                print(f"    已服务(卡车路径客户) NODE_ID: {served_nids}")
                print(f"    未服务(卡车路径客户) NODE_ID: {remaining_nids}")

            drone_served = get_drone_served_before_t(full_b2d_cur, full_finish_cur, decision_time)
            drone_served_uniq = sorted(set(drone_served))
            if verbose:
                print("    已服务(无人机) NODE_ID:", [data_cur.nodes[i]['node_id'] for i in drone_served_uniq])

            all_customers = {i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'customer'}

            # 【关键修复】卡车已服务集合必须以 split 的 served_nodes 为准，避免与 remaining_nodes 冲突导致漏点
            truck_served = set(served_nodes)

            drone_served_set = set(get_drone_served_before_t(full_b2d_cur, full_finish_cur, decision_time))
            served_all = truck_served | drone_served_set
            unserved_all = all_customers - served_all
            # 自检：split 给出的 served/remaining 必须互斥（否则时间轴有bug）
            _inter = set(served_nodes) & set(remaining_nodes)
            if _inter:
                raise RuntimeError(f"[BUG][SPLIT] served_nodes 与 remaining_nodes 重叠: {_inter}")

            if verbose:
                print(
                    f"    已完成(全系统) 客户数: {len(served_all)} = truck {len(truck_served)} + drone {len(drone_served_set)} (去重后)")
                print(f"    未完成(全系统) 客户数: {len(unserved_all)}")
                print("    未完成(全系统) NODE_ID:", [data_cur.nodes[i]['node_id'] for i in sorted(unserved_all)])

            # ---------- 5.2 基站集合（语义区分：future vs feasible） ----------
            all_bases = [i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'base']
            if data_cur.central_idx not in all_bases:
                all_bases.append(data_cur.central_idx)
            route_set_cur = set(full_route_cur)
            bases_in_plan = [b for b in all_bases if (b in route_set_cur) or (b == data_cur.central_idx)]
            visited_bases = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) <= decision_time + 1e-9]
            bases_to_visit = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) > decision_time + 1e-9]
            feasible_bases_for_drone = sorted(set(visited_bases + bases_to_visit))
            arrival_prefix = {b: full_arrival_cur[b] for b in visited_bases if b in full_arrival_cur}
            if verbose:
                print(f'    [bases_to_visit] t={decision_time:.2f}h => {[data_cur.nodes[b]["node_id"] for b in bases_to_visit]}')

            # ---------- 5.3 应用位置变更（events.csv：仅事实；ACCEPT/REJECT 由算法决定） ----------
            client_to_base_cur = build_client_to_base_map(full_b2d_cur)

            req_override = []
            predefined_xy = {}
            predefined_types = {}
            predefined_delta_avail = {}
            decisions_raw = []
            req_clients = []

            key = round(float(decision_time), 6)
            evs = offline_groups.get(key, []) if offline_groups is not None else []
            _ev_meta = {}
            for e in evs:
                nid = int(e.get('NODE_ID', 0))
                cidx = nodeid2idx.get(nid, None)
                if cidx is None:
                    continue
                # served_all 在本轮 5.1 后已计算
                if 'served_all' in locals() and (int(cidx) in served_all):
                    decision_log_rows.append({
                        'EVENT_ID': int(e.get('EVENT_ID', 0)),
                        'EVENT_TIME': float(e.get('EVENT_TIME', decision_time)),
                        'NODE_ID': nid,
                        'DECISION': 'EXPIRED',
                        'REASON': '决策点已服务，事件过期',
                        'OLD_X': float(data_cur.nodes[cidx].get('x', data_cur.nodes[cidx].get('orig_x', 0.0))),
                        'OLD_Y': float(data_cur.nodes[cidx].get('y', data_cur.nodes[cidx].get('orig_y', 0.0))),
                        'NEW_X': float(e.get('NEW_X', 0.0)),
                        'NEW_Y': float(e.get('NEW_Y', 0.0)),
                        'EVENT_CLASS': str(e.get('EVENT_CLASS', '')),
                        'APPLIED_X': '',
                        'APPLIED_Y': '',
                        'FORCE_TRUCK': '',
                        'BASE_LOCK': ''
                    })
                    continue
                req_override.append(int(cidx))
                predefined_xy[int(cidx)] = (float(e.get('NEW_X', 0.0)), float(e.get('NEW_Y', 0.0)))
                predefined_delta_avail[int(cidx)] = float(e.get('DELTA_AVAIL_H', 0.0))
                predefined_types[int(cidx)] = map_event_class_to_reloc_type(e.get('EVENT_CLASS', ''))
                _ev_meta[int(cidx)] = {
                    'EVENT_ID': int(e.get('EVENT_ID', 0)),
                    'EVENT_CLASS': str(e.get('EVENT_CLASS', '')),
                    'DELTA_AVAIL_H': float(e.get('DELTA_AVAIL_H', 0.0))
                }

            # 去重（保持稳定顺序）
            seen = set()
            req_override = [c for c in req_override if (c not in seen and not seen.add(c))]

            if verbose:
                print(f"    [REQ-OFFLINE] t={decision_time:.2f}h events={len(evs)} active={len(req_override)}")

            if len(req_override) > 0:
                # 将本决策点事件 meta 暂存到 data_cur（仅用于落盘 decision_log_rows）
                data_cur._offline_ev_meta = _ev_meta
                data_prelim, decisions_raw, req_clients = apply_relocations_for_decision_time(
                    data_cur, t_prev, decision_time,
                    full_depart_cur, full_finish_cur, full_arrival_cur,
                    client_to_base_cur,
                    req_override, predefined_xy, predefined_types, predefined_delta_avail
                )

                # ---------- [OFFLINE] 将本决策点的 ACCEPT/REJECT 结果写入 decision_log_rows ----------
                _meta = {}
                try:
                    _meta = getattr(data_cur, "_offline_ev_meta", {}) or {}
                except Exception:
                    _meta = {}
                if False and _meta and decisions_raw:
                    for _tup in decisions_raw:
                        try:
                            _cidx, _nid, _dec, _reason, _ox, _oy, _nx, _ny = _tup[:8]
                        except Exception:
                            continue
                        _m = _meta.get(int(_cidx), {})
                        _eid = _m.get("EVENT_ID", "")
                        _ecls = _m.get("EVENT_CLASS", "")
                        _apx = _nx if str(_dec).upper() == "ACCEPT" else _ox
                        _apy = _ny if str(_dec).upper() == "ACCEPT" else _oy
                        decision_log_rows.append({
                            "EVENT_ID": _eid,
                            "EVENT_TIME": float(decision_time),
                            "NODE_ID": int(_nid),
                            "DECISION": str(_dec),
                            "REASON": str(_reason),
                            "OLD_X": float(_ox),
                            "OLD_Y": float(_oy),
                            "NEW_X": float(_nx),
                            "NEW_Y": float(_ny),
                            "EVENT_CLASS": str(_ecls),
                            "APPLIED_X": float(_apx),
                            "APPLIED_Y": float(_apy),
                            "FORCE_TRUCK": "",
                            "BASE_LOCK": ""
                        })
                # 清空，避免跨决策点污染
                try:
                    data_cur._offline_ev_meta = {}
                except Exception:
                    pass
            else:
                data_prelim = data_cur

            # ---------- 5.4 判断是否存在尚未完成的无人机任务（用于 early-stop 口径） ----------
            unfinished_drone_exist = any(
                (full_finish_cur.get(_c, float('inf')) > decision_time + 1e-9) for _c in list(full_finish_cur.keys())
            )
            if (len(unserved_all) == 0) and (not unfinished_drone_exist) and (len(req_clients) == 0):
                if verbose:
                    print(
                        f"    [EARLY-STOP] t={decision_time:.2f}h：无未服务客户、无未完成无人机任务、且无新请求，提前结束后续决策点。")
                break
            # ---------- 决策点无新请求：不进行路径重规划（继续沿用原计划，等待后续决策点） ----------
            if len(req_clients) == 0:
                if verbose:
                    print(f"    [SKIP-REPLAN] t={decision_time:.2f}h：本决策点无新请求，不触发重规划。")
                t_prev = decision_time
                scene_idx += 1
                continue

            for d in decisions_raw:

                if verbose:
                    print("  ", d)
            # ===== 2.1) 快速筛选：把“代价太大”的 ACCEPT 改为 REJECT =====
            DELTA_COST_MAX = 30.0
            DELTA_LATE_MAX = 0.10

            data_next, decisions, qf_deltas = quick_filter_relocations(
                data_cur=data_cur,
                data_prelim=data_prelim,
                full_route_cur=full_route_cur,  # 注意：用当前场景“完整计划”作为快速评估基线
                full_b2d_cur=full_b2d_cur,
                prefix_route=prefix_route,
                req_clients=req_clients,
                decisions=decisions_raw,
                alpha_drone=0.3,
                lambda_late=50.0,
                truck_speed=TRUCK_SPEED_UNITS,
                drone_speed=DRONE_SPEED_UNITS,
                delta_cost_max=DELTA_COST_MAX,
                delta_late_max=DELTA_LATE_MAX,
            )

            # ---------- [OFFLINE] 将最终 ACCEPT/REJECT（含Δ指标与承诺/有效截止时间）写入 decision_log_rows ----------
            try:
                for (cid, dec, nx, ny, reason) in decisions:
                    cid = int(cid)
                    meta = _ev_meta.get(cid, {}) if isinstance(_ev_meta, dict) else {}
                    _eid = int(meta.get('EVENT_ID', -1))
                    _ecls = str(meta.get('EVENT_CLASS', ''))
                    _dav = float(meta.get('DELTA_AVAIL_H', predefined_delta_avail.get(cid, 0.0) if 'predefined_delta_avail' in locals() else 0.0))

                    # old 坐标来自 data_cur（apply 使用深拷贝，不会污染 data_cur）
                    _ox = float(data_cur.nodes[cid].get('x', 0.0))
                    _oy = float(data_cur.nodes[cid].get('y', 0.0))

                    # applied 坐标来自 data_next（quick_filter 最终状态）
                    _apx = float(data_next.nodes[cid].get('x', _ox))
                    _apy = float(data_next.nodes[cid].get('y', _oy))

                    pr = float(data_next.nodes[cid].get('prom_ready', data_next.nodes[cid].get('ready_time', 0.0)))
                    pd = float(data_next.nodes[cid].get('prom_due', data_next.nodes[cid].get('due_time', 0.0)))
                    ed = float(data_next.nodes[cid].get('due_time', pd))

                    drec = qf_deltas.get(cid, {}) if isinstance(qf_deltas, dict) else {}
                    d_cost = float(drec.get('D_COST', 0.0))
                    d_lp = float(drec.get('D_LATE_PROM', 0.0))
                    d_le = float(drec.get('D_LATE_EFF', 0.0))

                    decision_log_rows.append({
                        'EVENT_ID': _eid,
                        'EVENT_TIME': float(decision_time),
                        'NODE_ID': int(data_cur.nodes[cid].get('node_id', cid)),
                        'DECISION': str(dec),
                        'REASON': str(reason),
                        'OLD_X': _ox,
                        'OLD_Y': _oy,
                        'NEW_X': float(nx),
                        'NEW_Y': float(ny),
                        'EVENT_CLASS': str(_ecls),
                        'APPLIED_X': _apx,
                        'APPLIED_Y': _apy,
                        'FORCE_TRUCK': int(data_next.nodes[cid].get('force_truck', 0)),
                        'BASE_LOCK': ('' if data_next.nodes[cid].get('base_lock', None) is None else int(data_next.nodes[cid].get('base_lock'))),
                        'DELTA_AVAIL_H': _dav,
                        'PROM_READY': pr,
                        'PROM_DUE': pd,
                        'EFFECTIVE_DUE': ed,
                        'D_COST': d_cost,
                        'D_LATE_PROM': d_lp,
                        'D_LATE_EFF': d_le,
                    })
            except Exception as _e:
                print(f"[OFFLINE-LOG-WARN] write decision_log_rows failed: {_e}")



            # 兜底强制卡车直送集合（关键：先算它，后面 allowed 要用）
            forced_truck = {i for i in data_next.customer_indices
                            if data_next.nodes[i].get('force_truck', 0) == 1}
            # req_set：只让“未完成客户”进入后续重规划集合（关键修复）
            req_all = {int(c) for c in req_clients}
            req_set = {c for c in req_all if c in unserved_all}  # 只保留未完成
            req_served = req_all - req_set  # 已完成但提出请求的客户

            if req_served:
                print("[GUARD] 已完成客户提出变更请求(将不参与重规划):",
                      [data_next.nodes[i]["node_id"] for i in sorted(req_served)])

            print("本次决策的客户变更结果：")
            if not decisions:
                print("  无客户在该时刻提出变更请求，或全部不满足条件。")
            else:
                for it in decisions:
                    # quick_filter 后 decisions 为 (cid, dec, nx, ny, reason)
                    if isinstance(it, (list, tuple)) and len(it) == 8:
                        c, nid, dec, reason, ox, oy, nx, ny = it
                        print(f"  client_idx={c}, node_id={nid}, decision={dec}, "
                              f"old=({ox:.2f},{oy:.2f}), new=({nx:.2f},{ny:.2f}), 原因: {reason}")
                    elif isinstance(it, (list, tuple)) and len(it) == 5:
                        c, dec, nx, ny, reason = it
                        nid = data_cur.nodes[int(c)]['node_id']
                        ox = data_cur.nodes[int(c)]['x']
                        oy = data_cur.nodes[int(c)]['y']
                        print(f"  client_idx={c}, node_id={nid}, decision={dec}, "
                              f"old=({ox:.2f},{oy:.2f}), new=({nx:.2f},{ny:.2f}), 原因: {reason}")
                    else:
                        print(f"  [WARN] decisions 格式异常: {it}")

            # ---------- 护栏：无有效变更时跳过重规划（只推进时间窗口） ----------
            # 说明：如果本决策窗口内没有任何 ACCEPT（即没有坐标/模式变更落地），且没有新增 force_truck，
            # 则继续沿用 full_route_cur/full_b2d_cur，不做 classify + ALNS（节省算力，且更符合“无事件不重规划”）。
            num_req = len(decisions)
            num_acc = sum(1 for d in decisions if _decision_tag(d).startswith("ACCEPT"))
            num_rej = sum(1 for d in decisions if _decision_tag(d).startswith("REJECT"))
            if (num_acc == 0) and (len(forced_truck) == 0):
                if verbose:
                    print(f"    [SKIP-REPLAN] t={decision_time:.2f}h：无ACCEPT请求且无force_truck变更，跳过路径重规划，沿用当前全局计划。")
                # 仍然记录该场景指标（全局口径，不重规划）
                full_eval_skip = sim.evaluate_full_system(
                    data_next, full_route_cur, full_b2d_cur,
                    alpha_drone=0.3, lambda_late=50.0,
                    truck_speed=TRUCK_SPEED_UNITS, drone_speed=DRONE_SPEED_UNITS
                )
                scenario_results.append(
                    _pack_scene_record(scene_idx, decision_time, full_eval_skip,
                                       num_req=num_req, num_acc=num_acc, num_rej=num_rej,
                                       alpha_drone=0.3, lambda_late=50.0)
                )

                # [PROMISE] 场景日志输出（skip-replan）
                _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
                emit_scene_late_logs(_late_dir, scene_idx=scene_idx, decision_time=decision_time, data=data_next, full_route=full_route_cur, full_b2d=full_b2d_cur, full_eval=full_eval_skip, prefix="", drone_speed=DRONE_SPEED_UNITS)
                # 推进窗口，进入下一决策点（不改变 full_* 计划）
                data_cur = data_next
                t_prev = decision_time
                scene_idx += 1
                continue

            # ---------- 5.4 重规划起点（虚拟节点） ----------
            if virtual_pos is not None:
                start_idx_for_alns = add_virtual_truck_position_node(data_next, virtual_pos)
            else:
                start_idx_for_alns = current_node

            # 兜底修正：避免从仓库 0 重新起步（注意：这里绝不覆盖 t_prev）
            if (start_idx_for_alns == data_next.central_idx
                    and decision_time > 1e-6
                    and len(remaining_nodes) > 0):

                first_node = remaining_nodes[0]
                seg_t0 = full_arrival_cur.get(data_cur.central_idx, 0.0)
                seg_t1 = full_arrival_cur.get(first_node, seg_t0 + 1e-6)

                if seg_t1 > seg_t0 + 1e-9:
                    ratio = (decision_time - seg_t0) / (seg_t1 - seg_t0)
                    ratio = max(0.0, min(1.0, ratio))

                    x0 = data_next.nodes[data_next.central_idx]['x']
                    y0 = data_next.nodes[data_next.central_idx]['y']
                    x1 = data_next.nodes[first_node]['x']
                    y1 = data_next.nodes[first_node]['y']

                    x_cur = x0 + ratio * (x1 - x0)
                    y_cur = y0 + ratio * (y1 - y0)

                    start_idx_for_alns = add_virtual_truck_position_node(data_next, (x_cur, y_cur))
                    dprint(
                        f"  [fix] 起点从仓库 0 修正为虚拟位置 idx={start_idx_for_alns}, pos=({x_cur:.2f},{y_cur:.2f})")

            # ---------- 5.5 构造 allowed_customers（这是你之前最容易乱的地方） ----------
            remaining_truck_route_nodes = set(remaining_nodes)

            unfinished_drone = set()
            for b, clients in full_b2d_cur.items():
                t_base = full_arrival_cur.get(b, float('inf'))
                for c in clients:
                    if full_finish_cur.get(c, float('inf')) <= decision_time + 1e-9:
                        continue
                    if t_base > decision_time + 1e-9:
                        unfinished_drone.add(c)
                    else:
                        if full_depart_cur.get(c, float('inf')) > decision_time + 1e-9:
                            unfinished_drone.add(c)
            # ---------- 修复：force_truck 只对“未完成客户”生效 ----------
            forced_truck = set(forced_truck)

            # 1) 已服务的 force_truck 必须剔除，否则与 served_guard 冲突
            served_in_force = forced_truck & served_all
            if served_in_force:
                if verbose:
                    print(f"[FIX] force_truck 中包含已服务客户，已剔除：{sorted(served_in_force)}")
                forced_truck -= served_in_force
            # ====== 强制对齐：卡车 served/remaining 以 split 输出为唯一真值（放在 allowed_customers 构造前）======
            truck_served = set(served_nodes)  # t_dec 前卡车已服务客户集合
            remaining_truck_route_nodes = set(remaining_nodes)  # t_dec 后卡车后缀仍需考虑的客户集合

            # 自检：二者必须互斥，否则后续会出现“本该重规划却被当 served 剔除”的漏点
            _inter = truck_served & remaining_truck_route_nodes
            if _inter:
                if verbose:
                    print("[BUG][SPLIT] truck_served 与 remaining 重叠：", sorted(_inter))
                raise RuntimeError("[BUG][SPLIT] served/remaining 不互斥，请检查 split 逻辑或时间轴")

            # 2) 用过滤后的 forced_truck 构造 allowed
            allowed_customers = (remaining_truck_route_nodes | unfinished_drone | req_set | forced_truck)

            # 护栏：已完成客户绝不允许进入 allowed（否则会被重新规划/重复服务）
            served_in_allowed = allowed_customers & served_all
            if served_in_allowed:
                if verbose:
                    print(f"[GUARD][FIX] served_all ∩ allowed_customers 非空，已自动剔除： {sorted(served_in_allowed)}")
                allowed_customers -= served_in_allowed

            # 护栏：force_truck 必须进 allowed（这里 forced_truck 已经剔除了 served，所以不再矛盾）
            missing = forced_truck - allowed_customers
            if missing:
                raise RuntimeError(f"[GUARD] force_truck 客户未进入 allowed_customers: node_id={sorted(missing)}")

            # ====== 5.A 构造结构敏感集合 ======
            C_moved_accept, C_moved_reject = set(), set()
            for it in decisions:
                cid = int(it[0])
                if cid not in unserved_all:
                    continue  # 已完成客户不进入结构敏感集合（否则算子会“扰动已完成点”）
                dec = str(it[1])
                if dec.startswith("ACCEPT"):
                    C_moved_accept.add(cid)
                elif dec.startswith("REJECT"):
                    C_moved_reject.add(cid)

            C_force_truck = {i for i in data_next.customer_indices if data_next.nodes[i].get("force_truck", 0) == 1}

            # 覆盖边界：只在 allowed_customers 中挑 top-k
            def _boundary_score(i):
                xi, yi = data_next.nodes[i]["x"], data_next.nodes[i]["y"]
                best = float("inf")
                for b in feasible_bases_for_drone:
                    xb, yb = data_next.nodes[b]["x"], data_next.nodes[b]["y"]
                    d = ((xi - xb) ** 2 + (yi - yb) ** 2) ** 0.5
                    best = min(best, abs(2 * d - sim.DRONE_RANGE_UNITS))
                return best

            k_boundary = 6
            cand_boundary = [i for i in allowed_customers
                             if data_next.nodes[i].get("node_type") == "customer"
                             and i not in C_force_truck]
            cand_boundary.sort(key=_boundary_score)
            C_boundary = set(cand_boundary[:k_boundary])

            print(f"[SET] moved_acc={len(C_moved_accept)} moved_rej={len(C_moved_reject)} "
                  f"force={len(C_force_truck)} boundary={len(C_boundary)}")

            # ---------- 5.6 分类（只在 allowed 内重新分配 truck/drone） ----------
            base_to_drone_next, truck_next = sim.classify_clients_for_drone(
                data_next,
                allowed_customers=allowed_customers,
                feasible_bases=feasible_bases_for_drone
            )

            drone_assigned = {c for cs in base_to_drone_next.values() for c in cs}

            # 护栏：force_truck 不允许进无人机
            bad = forced_truck & drone_assigned
            if bad:
                raise RuntimeError(f"[GUARD] force_truck 客户被分配给无人机: node_id="
                                   f"{[data_next.nodes[i]['node_id'] for i in bad]}")

            # ---------- 5.7 放宽无人机时间窗（对新分配生效） ----------
            # [PROMISE] 场景0/动态阶段不再对无人机时间窗做 relax/reset（承诺窗由场景0 ETA0 冻结生成）
            if False:
                            relax_drone_time_windows(
                                data_next, base_to_drone_next,
                                extra_slack=2.0,
                                reset_ready_to_zero=True
                            )


            # ---------- 5.8 ALNS 重规划（后缀） ----------
            ctx_for_alns = dict(ab_cfg)  # 关键：先注入消融配置（paired/free/算子池）
            ctx_for_alns["verbose"] = verbose
            ctx_for_alns.update({
                "C_moved_accept": C_moved_accept,
                "C_moved_reject": C_moved_reject,
                "C_force_truck": C_force_truck,
                "C_boundary": C_boundary,
                "min_remove": 3,  # 动态场景至少移除3个点，保证有“重构力度”
            })

            # 基站语义区分：用于无人机可行基站与到达时间合并（关键修复）
            ctx_for_alns["feasible_bases_for_drone"] = feasible_bases_for_drone
            ctx_for_alns["visited_bases"] = set(visited_bases)
            ctx_for_alns["arrival_prefix"] = arrival_prefix


            (route_next, b2d_next, cost_next, truck_dist_next, drone_dist_next, total_late_next,
             truck_time_next) = alns_truck_drone(data_next, base_to_drone_next, max_iter=1000,
                                                 remove_fraction=0.1, T_start=1.0, T_end=0.01, alpha_drone=0.3,
                                                 lambda_late=50.0,
                                                 truck_customers=truck_next, use_rl=False, rl_tau=0.5, rl_eta=0.1,
                                                 start_idx=start_idx_for_alns,
                                                 start_time=decision_time, bases_to_visit=bases_to_visit,
                                                 ctx=ctx_for_alns)

            if b2d_next is None:
                raise RuntimeError("alns_truck_drone 返回 b2d_next=None，请检查函数返回值。")

            # 本场景后缀时间表（从 decision_time 开始）
            arrival_next, total_time_next, _ = sim.compute_truck_schedule(
                data_next, route_next, start_time=decision_time, speed=TRUCK_SPEED_UNITS
            )
            arrival_next_merged = dict(arrival_prefix)
            arrival_next_merged.update(arrival_next)
            depart_next, finish_next, base_finish_next = compute_multi_drone_schedule(
                data_next, b2d_next, arrival_next_merged,
                num_drones_per_base=NUM_DRONES_PER_BASE,
                drone_speed=DRONE_SPEED_UNITS
            )
            system_finish_next = max(
                total_time_next,
                max(base_finish_next.values()) if base_finish_next else 0.0
            )
            sim.check_disjoint(data_next, route_next, b2d_next)

            print(f"场景 {scene_idx}: 成本={cost_next:.3f}, 卡车距={truck_dist_next:.3f}, "
                  f"无人机距={drone_dist_next:.3f}, 总迟到={total_late_next:.3f}, "
                  f"卡车总时间={truck_time_next:.2f}h, 系统完成时间={system_finish_next:.2f}h")

            num_req = len(decisions)
            num_acc = sum(1 for d in decisions if _decision_tag(d).startswith("ACCEPT"))
            num_rej = sum(1 for d in decisions if _decision_tag(d).startswith("REJECT"))

            # ---------- 5.9 拼接“全局完整解”（FULL） ----------
            suffix_for_full = route_next[:]
            if suffix_for_full and (
                    data_next.nodes[suffix_for_full[0]].get('node_type') == 'truck_pos'
                    or data_next.nodes[suffix_for_full[0]].get('node_id') == -1
            ):
                suffix_for_full = suffix_for_full[1:]

            full_route_next = _merge_prefix_suffix(prefix_route, suffix_for_full)

            full_b2d_next = {b: cs.copy() for b, cs in full_b2d_cur.items()}

            # 从历史无人机任务中移除 allowed（避免重复）
            allowed_set = set(allowed_customers)
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in allowed_set]

            # 合并本场景新分配
            for b, cs in b2d_next.items():
                full_b2d_next.setdefault(b, []).extend(cs)

            # 再移除 force_truck（兜底）
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in forced_truck]

            # 全局去重：一客一基站
            seen = set()
            for b in list(full_b2d_next.keys()):
                new_list = []
                for c in full_b2d_next[b]:
                    if c in seen:
                        continue
                    seen.add(c)
                    new_list.append(c)
                full_b2d_next[b] = new_list

            # 卡车优先：去掉与卡车路径冲突的无人机客户
            truck_nodes_set = {
                i for i in full_route_next
                if 0 <= i < len(data_next.nodes) and data_next.nodes[i].get('node_type') == 'customer'
            }
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in truck_nodes_set]
            full_route_next = cover_uncovered_by_truck_suffix(
                data_next, full_route_next, full_b2d_next, prefix_len=len(prefix_route), verbose=verbose
            )
            # 如果你有 guardrail_check 就保留，没有就删
            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" after-merge scene={scene_idx}")

            # FULL 全局时间轴重算（从0开始）
            full_arrival_next, full_total_time_next, full_truck_late_next = sim.compute_truck_schedule(
                data_next, full_route_next, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
            )
            full_depart_next, full_finish_next, full_base_finish_next = compute_multi_drone_schedule(
                data_next, full_b2d_next, full_arrival_next,
                num_drones_per_base=NUM_DRONES_PER_BASE,
                drone_speed=DRONE_SPEED_UNITS
            )
            # ---------- 统一 finish_time 口径：卡车客户=truck_arrival，无人机客户=drone_finish ----------
            full_finish_all_next = dict(full_arrival_next)  # 先用卡车到达时间填满（包含所有卡车路径节点）

            # 用无人机完成时间覆盖对应客户（只覆盖 customer 节点）
            for _cid, _fin in full_finish_next.items():
                try:
                    _cid_int = int(_cid)
                except Exception:
                    continue
                if 0 <= _cid_int < len(data_next.nodes) and str(
                        data_next.nodes[_cid_int].get("node_type", "")).lower() == "customer":
                    full_finish_all_next[_cid_int] = float(_fin)

            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" after-schedule scene={scene_idx}")

            full_eval = sim.evaluate_full_system(
                data_next, full_route_next, full_b2d_next,
                alpha_drone=0.3, lambda_late=50.0,
                truck_speed=TRUCK_SPEED_UNITS, drone_speed=DRONE_SPEED_UNITS
            )

            # [PROMISE] 场景日志输出（replan）
            _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
            emit_scene_late_logs(_late_dir, scene_idx=scene_idx, decision_time=decision_time, data=data_next, full_route=full_route_next, full_b2d=full_b2d_next, full_eval=full_eval, prefix="", drone_speed=DRONE_SPEED_UNITS)

            check_info = sim.sanity_check_full(data_next, full_route_next, full_b2d_next)
            print("[FULL-check]", check_info)
            print(f"[FULL] cost={full_eval['cost']:.3f}, system_time={full_eval['system_time']:.3f}h, "
                  f"late={full_eval['total_late']:.3f} (truck={full_eval['truck_late']:.3f}, drone={full_eval['drone_late']:.3f}), "
                  f"truck_dist={full_eval['truck_dist']:.3f}, drone_dist={full_eval['drone_dist']:.3f}")

            scenario_results.append({
                "scene": scene_idx,
                "t_dec": decision_time,
                "cost": full_eval["cost"],
                "base_cost": full_eval.get("truck_dist_eff", full_eval["truck_dist"]) + 0.3 * full_eval["drone_dist"],
                "penalty": 50.0 * full_eval["total_late"],
                "lambda_late": 50.0,
                "truck_dist": full_eval["truck_dist"],
                "drone_dist": full_eval["drone_dist"],
                "system_time": full_eval["system_time"],
                "truck_late": full_eval["truck_late"],
                "drone_late": full_eval["drone_late"],
                "total_late": full_eval["total_late"],
                "num_req": num_req,
                "num_acc": num_acc,
                "num_rej": num_rej
            })

            # 画图：对比前一轮无人机集合（可选）
            drone_set_before = set()
            for cs in full_b2d_cur.values():
                drone_set_before.update(cs)

            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" before-plot scene={scene_idx}")

                        # 兼容：visualize_truck_drone 需要 8 元组 (cidx, nid, dec, reason, ox, oy, nx, ny)
            decisions_viz = _normalize_decisions_for_viz(data_cur, decisions)
            if enable_plot:
                visualize_truck_drone(
                    data_next,
                    full_route_next,
                    full_b2d_next,
                    title=f"Scenario {scene_idx}: progress view (t={decision_time:.2f}h)",
                    xlim=global_xlim,
                    ylim=global_ylim,
                    decision_time=decision_time,
                    truck_arrival=full_arrival_next,
                    drone_finish=full_finish_next,
                    prefix_route=prefix_route_for_plot,
                    virtual_pos=virtual_pos,
                    relocation_decisions=decisions_viz,
                    drone_set_before=drone_set_before
                )

            # ---------- 5.10 更新滚动状态，进入下一决策点 ----------
            data_cur = data_next
            full_route_cur = full_route_next
            full_b2d_cur = full_b2d_next
            full_arrival_cur = full_arrival_next
            full_depart_cur = full_depart_next
            full_finish_cur = full_finish_all_next

            t_prev = decision_time
            scene_idx += 1

    # ===================== 6) 汇总输出（run_one 里建议不打印，交给 main 或 ablation）=====================

    # ---------- 保存 decision_log（离线 events.csv 模式） ----------
    try:
        if offline_groups is not None:
            _out = decision_log_path
            if (not _out) or (str(_out).strip() == ""):
                base = os.path.splitext(os.path.basename(file_path))[0]
                _dir = os.path.dirname(events_path) if events_path else os.path.dirname(file_path)
                _out = os.path.join(_dir or ".", f"decision_log_{base}_seed{seed}.csv")
            save_decision_log(decision_log_rows, _out)
            if verbose:
                print(f"[LOG] decision log saved: {_out} (rows={len(decision_log_rows)})")
    except Exception as _e:
        if verbose:
            print("[WARN] decision_log 保存失败：", _e)

    return scenario_results
# ===================== 纯卡车 vs 卡车-无人机（静态距离/成本对比）=====================
def print_summary_table(scenario_results):
    print("\n===== 动态位置变更场景汇总 =====")
    print("scene | t_dec | cost(obj) | base_cost | penalty | truck_dist | drone_dist | system_time | total_late | req | acc | rej")
    for rec in scenario_results:
        base_cost = rec.get("base_cost", None)
        penalty = rec.get("penalty", None)

        # 兼容旧结果：若未提供拆分，则用 cost 与 total_late 反推（前提：lambda_late 在 rec 或使用默认 50.0）
        if base_cost is None or penalty is None:
            lam = float(rec.get("lambda_late", 50.0))
            try:
                penalty = lam * float(rec["total_late"])
                base_cost = float(rec["cost"]) - penalty
            except Exception:
                penalty = 0.0
                base_cost = float(rec.get("cost", 0.0))

        print(f"{rec['scene']:5d} | "
              f"{rec['t_dec']:5.2f} | "
              f"{rec['cost']:8.3f} | "
              f"{base_cost:8.3f} | "
              f"{penalty:7.3f} | "
              f"{rec['truck_dist']:10.3f} | "
              f"{rec['drone_dist']:10.3f} | "
              f"{rec['system_time']:11.3f} | "
              f"{rec['total_late']:9.3f} | "
              f"{rec['num_req']:3d} | "
              f"{rec['num_acc']:3d} | "
                      f"{rec['num_rej']:3d}")
        # # --- 货币口径（元）：用于论文/对比（不影响 obj 计算） ---
        # try:
        #     c_truck_km = float(rec.get("c_truck_km", 1.0))
        #     alpha = float(rec.get("alpha_drone", 0.3))
        #     c_drone_km = float(rec.get("c_drone_km", alpha * c_truck_km))
        #     truck_km_euclid = float(rec.get("truck_dist", 0.0)) / float(SCALE_KM_PER_UNIT)
        #     # 关键口径：truck_km 表示“已包含路况系数(绕行)后的实际卡车里程（km）”
        #     truck_km = truck_km_euclid  # 中文注释：rec['truck_dist'] 已包含路况系数，不要二次乘
        #     drone_km = float(rec.get("drone_dist", 0.0)) / float(SCALE_KM_PER_UNIT)
        #     base_money = c_truck_km * truck_km + c_drone_km * drone_km
        #     lam = float(rec.get("lambda_late", 50.0))
        #     penalty_money = lam * float(rec.get("total_late", 0.0))
        #     cost_money = base_money + penalty_money
        #     print(
        #         f"      money: cost={cost_money:.3f} base={base_money:.3f} penalty={penalty_money:.3f} "
        #         f"(truck_km(road)={truck_km:.3f}, drone_km={drone_km:.3f}, c_truck={c_truck_km:.3f}, c_drone={c_drone_km:.3f})"
        #     )
        # except Exception:
        #     pass

def _resolve_operator_list(names, g):
    """把字符串函数名解析成函数对象；找不到就报明确错误。"""
    ops = []
    for nm in names:
        fn = g.get(nm, None)
        if fn is None or not callable(fn):
            raise RuntimeError(f"[CFG] 未找到算子函数: {nm}")
        ops.append(fn)
    return ops



def _print_operator_stats(op_stat: dict, top_k: int = 20):
    """打印 ALNS 算子统计（calls/accepts/sa_reject/late_fail 等），用于验证消融是否真起作用。"""
    if not op_stat:
        print("[ALNS-SUM] op_stat empty")
        return
    rows = []
    for k, s in op_stat.items():
        calls = int(s.get("calls", 0))
        acc = int(s.get("accepts", 0))
        sarej = int(s.get("sa_reject", 0))
        latef = int(s.get("late_fail", 0))
        latedf = int(s.get("late_delta_fail", 0))
        repf = int(s.get("repair_fail", 0))
        covf = int(s.get("cover_fail", 0))
        besth = int(s.get("best_hits", 0))
        bestg = float(s.get("best_gain", 0.0))
        acc_rate = (acc / calls) if calls > 0 else 0.0
        best_rate = (besth / calls) if calls > 0 else 0.0
        rows.append((k, calls, acc, acc_rate, besth, best_rate, sarej, latef, latedf, repf, covf, bestg))
    rows.sort(key=lambda x: (-x[3], -x[5], -x[11]))
    print("[ALNS-SUM] op_key | calls acc acc_rate best_hits best_rate sa_rej late_abs_fail late_delta_fail repair_fail cover_fail best_gain")
    for r in rows[:top_k]:
        print("  - %s | %4d %4d %.2f %4d %.2f %4d %4d %4d %4d %4d %.3f" %
              (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]))
def build_ab_cfg(cfg: dict):
    """把 cfg 中的字符串 DESTROYS/REPAIRS/ALLOWED_PAIRS 转成可执行函数对象。

    兼容两种写法：
      1) 传字符串函数名（推荐，便于日志/复现实验）
      2) 直接传函数对象（你当前 CFG_A 的写法）
    """
    new_cfg = dict(cfg)

    # DESTROYS / REPAIRS
    if "DESTROYS" in new_cfg and new_cfg["DESTROYS"]:
        if isinstance(new_cfg["DESTROYS"][0], str):
            new_cfg["DESTROYS"] = _resolve_operator_list(new_cfg["DESTROYS"], globals())

    if "REPAIRS" in new_cfg and new_cfg["REPAIRS"]:
        if isinstance(new_cfg["REPAIRS"][0], str):
            new_cfg["REPAIRS"] = _resolve_operator_list(new_cfg["REPAIRS"], globals())

    # ALLOWED_PAIRS（paired 模式）
    if "ALLOWED_PAIRS" in new_cfg and new_cfg["ALLOWED_PAIRS"]:
        pairs = []
        for dnm, rnm in new_cfg["ALLOWED_PAIRS"]:
            # destroy
            if isinstance(dnm, str):
                D = globals().get(dnm, None)
                if D is None or not callable(D):
                    raise RuntimeError(f"[CFG] 未找到 destroy: {dnm}")
            elif callable(dnm):
                D = dnm
            else:
                raise RuntimeError(f"[CFG] destroy 既不是字符串也不是函数对象: {dnm}")

            # repair
            if isinstance(rnm, str):
                R = globals().get(rnm, None)
                if R is None or not callable(R):
                    raise RuntimeError(f"[CFG] 未找到 repair: {rnm}")
            elif callable(rnm):
                R = rnm
            else:
                raise RuntimeError(f"[CFG] repair 既不是字符串也不是函数对象: {rnm}")

            pairs.append((D, R))
        new_cfg["ALLOWED_PAIRS"] = pairs

    return new_cfg


def _normalize_perturbation_times(times):
    """规范化决策时刻列表：
    - 过滤 <=0 的时刻（t=0 的初始场景由系统自动包含，避免重复）
    - 去重（按 1e-6 精度）
    - 升序排序
    """
    import math
    if not times:
        return []
    cleaned = []
    for t in times:
        try:
            ft = float(t)
        except Exception:
            continue
        if math.isnan(ft) or math.isinf(ft):
            continue
        if ft <= 1e-9:
            continue
        cleaned.append(round(ft, 6))
    return sorted(set(cleaned))



def run_static_truck_only(
        file_path: str,
        seed: int,
        ab_cfg: dict,
        enable_plot: bool = False,
        verbose: bool = True
):
    """静态-纯卡车基线（不使用无人机，不强制访问基站）。

    目的：和混合模式在“同一 ALNS 框架”下公平对比。
    做法：
    - 把所有 customer 标记 force_truck=1
    - base_to_drone_customers 置空
    - bases_to_visit 仅包含 central（避免初始路径把所有基站当作必访点）
    """
    if seed is not None:
        set_seed(int(seed))

    ab_cfg = build_ab_cfg(ab_cfg)

    data = read_data(file_path, scenario=0, strict_schema=True)

    # 1) 强制所有客户都走卡车
    truck_customers = list(getattr(data, "customer_indices", []))
    for cid in truck_customers:
        try:
            data.nodes[cid]["force_truck"] = 1
        except Exception:
            pass

    base_to_drone_customers = {}  # 纯卡车：无人机任务为空

    ctx0 = dict(ab_cfg)
    ctx0["verbose"] = verbose

    (best_route,
     best_b2d,
     _best_cost_internal,
     _best_truck_dist,
     _best_drone_dist,
     _best_total_late,
     _best_truck_time) = alns_truck_drone(
        data,
        base_to_drone_customers,
        max_iter=1000,
        remove_fraction=0.1,
        T_start=1.0,
        T_end=0.01,
        alpha_drone=0.3,
        lambda_late=50.0,
        truck_customers=truck_customers,
        use_rl=False,
        rl_tau=0.5,
        rl_eta=0.1,
        bases_to_visit=[data.central_idx],   # 关键：不强制访问所有基站
        ctx=ctx0
    )

    # 二次兜底：确保无人机为空、force_truck 不被破坏
    best_route, best_b2d = enforce_force_truck_solution(data, best_route, best_b2d)

    full_eval = sim.evaluate_full_system(
        data, best_route, best_b2d,
        alpha_drone=0.3, lambda_late=50.0,
        truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
    )

    rec = {
        "scene": 0,
        "t_dec": 0.0,
        "cost": full_eval["cost"],
        "base_cost": full_eval.get("truck_dist_eff", full_eval["truck_dist"]) + 0.3 * full_eval["drone_dist"],
        "penalty": 50.0 * full_eval["total_late"],
        "lambda_late": 50.0,
        "truck_dist": full_eval["truck_dist"],
        "drone_dist": full_eval["drone_dist"],
        "system_time": full_eval["system_time"],
        "truck_late": full_eval["truck_late"],
        "drone_late": full_eval["drone_late"],
        "total_late": full_eval["total_late"],
        "num_req": 0,
        "num_acc": 0,
        "num_rej": 0
    }

    if verbose:
        print("[TRUCK-ONLY] route NODE_ID:", [data.nodes[i].get("node_id", i) for i in best_route])
        print(f"[TRUCK-ONLY] cost={rec['cost']:.3f}, system_time={rec['system_time']:.3f}h, "
              f"late={rec['total_late']:.3f}, truck_dist={rec['truck_dist']:.3f}")

    if enable_plot:
        visualize_truck_drone(
            data, best_route, best_b2d,
            title="Static Truck-Only Baseline",
            show_base_radius=True,
            decisions=[],
            save_path=""
        )

    return rec

def main():
    """
    中文注释：主入口（不再使用命令行参数，所有实验参数集中在此处配置）。
    你只需要改下面这些变量即可复现：
    - file_path / events_path
    - seed / cfg / perturbation_times
    - road_factor（路况系数：只影响卡车弧距离=欧氏×系数，从而影响卡车时间/迟到与卡车距离成本）
    """
    print("[BOOT]", __file__, "DEBUG_LATE=", DEBUG_LATE, "DEBUG_LATE_SCENES=", DEBUG_LATE_SCENES)

    # ===== 1) 实验输入 =====
    file_path = r"D:\代码\ALNS+DL\nodes_25_seed2023_20260110_201842.csv"
    events_path = r"D:\代码\ALNS+DL\events_25_seed2023_20260110_201842.csv"   # 可选：离线事件脚本（仅事实，不含 accept/reject），为空则不使用
    seed = 2025
    cfg = CFG_D

    # 动态模式：决策点（小时），t=0 场景系统自动包含
    perturbation_times = [1.0, 2.0]

    enable_plot = True
    verbose = True

    # ===== 2) 路况系数（唯一入口：只放大卡车距离，不改速度）=====
    # 初始化仿真参数
    road_factor = 1.5
    sim.set_simulation_params(road_factor=road_factor)
    # 并且建议定义本地快捷变量，如果下面有用到
    TRUCK_SPEED_UNITS = sim.get_simulation_params()["TRUCK_SPEED_UNITS"]
    DRONE_SPEED_UNITS = sim.get_simulation_params()["DRONE_SPEED_UNITS"]
    NUM_DRONES_PER_BASE = sim.get_simulation_params()["NUM_DRONES_PER_BASE"]
    print(f"[PARAM] TRUCK_ROAD_FACTOR={sim.TRUCK_ROAD_FACTOR:.3f}; TRUCK_SPEED_UNITS={TRUCK_SPEED_UNITS:.3f} units/h (fixed); truck_arc = euclid * {sim.TRUCK_ROAD_FACTOR:.3f}")

    # ===== 3) 运行模式开关 =====
    # 3.1 静态对比：纯卡车 vs 混合（只跑 scene=0，不跑动态扰动）
    RUN_STATIC_COMPARE = False

    # 3.2 批量对比：多 seed + 多 CFG（使用同一 events.csv 即可保证同一请求流）
    RUN_BATCH = False

    # 3.3 最小可复现实验：road_factor 1.0 vs 1.5 应显著改变 system_time / late
    RUN_ROAD_SANITY = False

    # ===== 4) road sanity =====
    if RUN_ROAD_SANITY:
        for rf in [1.0, 1.5]:
            sim.set_simulation_params(rf)
            results = run_one(
                file_path=file_path, seed=seed, ab_cfg=cfg,
                perturbation_times=perturbation_times,
                enable_plot=False, verbose=False,
                events_path=events_path, decision_log_path=''
            )
            r0 = results[0]
            print(f"[SANITY] rf={rf:.1f} speed_units={TRUCK_SPEED_UNITS:.3f} truck_dist={r0['truck_dist']:.3f} drone_dist={r0['drone_dist']:.3f} system_time={r0['system_time']:.3f} total_late={r0['total_late']:.3f} cost={r0['cost']:.3f}")
        return

    # ===== 5) 静态纯卡车 vs 混合对比 =====
    if RUN_STATIC_COMPARE:
        # 只跑静态场景：不加任何决策点
        pert_static = []

        # 5.1 混合模式（原 run_one）
        sim.set_simulation_params(road_factor=road_factor)
        res_mixed = run_one(
            file_path=file_path, seed=seed, ab_cfg=cfg,
            perturbation_times=pert_static,
            enable_plot=False, verbose=False,
            events_path=events_path, decision_log_path=''
        )[0]

        # 5.2 纯卡车（不访问基站、不派无人机）
        sim.set_simulation_params(road_factor=road_factor)
        res_truck = run_static_truck_only(
            file_path=file_path, seed=seed, ab_cfg=cfg,
            enable_plot=False, verbose=False
        )

        print("\n===== STATIC COMPARE (scene=0) =====")
        print(f"[MIXED]      cost={res_mixed['cost']:.3f}  truck={res_mixed['truck_dist']:.3f}  drone={res_mixed['drone_dist']:.3f}  time={res_mixed['system_time']:.3f}  late={res_mixed['total_late']:.3f}")
        print(f"[TRUCK-ONLY] cost={res_truck['cost']:.3f}  truck={res_truck['truck_dist']:.3f}  drone={res_truck['drone_dist']:.3f}  time={res_truck['system_time']:.3f}  late={res_truck['total_late']:.3f}")
        return

    # ===== 6) 批量对比：多 seed + 多 cfg（不使用回放）=====
    if RUN_BATCH:
        cfgs = [CFG_A, CFG_D]
        seeds = [2021, 2022, 2023]  # 你也可以先用 [2025] 小跑验证
        enable_plot_batch = False
        verbose_batch = False
        # 公平性说明：
        # - events_path 非空：所有配置共享同一离线请求流（推荐/最公平）
        # - events_path 为空：仅运行场景0（无动态请求）；若要动态对比请提供 events_path
        for sd in seeds:
            for _cfg in cfgs:
                sim.set_simulation_params(road_factor=road_factor)
                res = run_one(
                    file_path=file_path, seed=sd, ab_cfg=_cfg,
                    perturbation_times=perturbation_times,
                    enable_plot=enable_plot_batch, verbose=verbose_batch,
                    events_path=events_path, decision_log_path=''
                )
                r0 = res[0]
                print(f"[BATCH] seed={sd} cfg={_cfg.get('name','?')} cost0={r0['cost']:.3f} time0={r0['system_time']:.3f} late0={r0['total_late']:.3f}")
        return
    
    # ===== 7) 正常动态运行（你平时跑的模式）=====

    results = run_one(
        file_path=file_path, seed=seed, ab_cfg=cfg,
        perturbation_times=perturbation_times,
        enable_plot=enable_plot, verbose=verbose,
        events_path=events_path, decision_log_path=''
    )
    print_summary_table(results)


if __name__ == "__main__":
    main()
