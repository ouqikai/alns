import copy
import math
import random
import os
import csv
import simulation as sim
import numpy as np
import utils as ut # apply函数依赖这个
from data_io_1322 import recompute_cost_and_nearest_base # 依赖这个重算距离
import operators as ops
import pandas as pd
import milp_solver as grb   # 你把 01162113.py 复制改名后的模块

# === 补充 dprint 定义 ===
DEBUG = False  # 如果你想看详细调试信息，改成 True

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
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

def _apply_xy(node, x, y, source="PERTURBED"):
    node["x"] = x
    node["y"] = y
    node["coord_source"] = source  # 关键：同步标记

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

def get_drone_served_before_t(full_b2d_cur, full_finish_cur, t, eps=1e-9):
    # t时刻之前已服务的无人机客户
    served = set()
    for b, cs in full_b2d_cur.items():
        for c in cs:
            if full_finish_cur.get(c, float('inf')) <= t + eps:
                served.add(c)
    return sorted(served)

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

def _rewrite_allowed_order_on_template(data, template_route, allowed_set, new_route, start_idx, depot_idx, bases_set):
    """
    中文注释（修复版）：
    - template_route：来自 preplan 的后缀模板（可能包含“冻结客户”与基站序）
    - new_route：ALNS/GRB 求出的后缀路线（允许在 allowed 内发生 truck/drone 模式变化）

    旧版风险：若 new_route 的“卡车客户集合”与 template_route 不一致（无人机->卡车 / 卡车->无人机），
    len(new_seq)!=len(pos) 就原样返回 => allowed 客户可能从无人机列表被删掉却没进卡车路线 => uncovered。

    新策略：
    1) 从 new_route 抽取“卡车要访问的 allowed customer 顺序” new_seq
    2) 扫描 template_route：
       - frozen（不在 allowed_set）客户：原样保留
       - allowed 客户：用 new_seq 依次替换；若 new_seq 不够，说明该客户被改为无人机服务 => 直接删除
    3) 若 new_seq 还有剩余（说明 new 解把原本无人机客户改为卡车服务），把剩余客户按顺序插入到 depot 前
    """
    if not template_route or not new_route:
        return list(template_route) if template_route is not None else []

    allowed_set = set(int(x) for x in allowed_set) if allowed_set is not None else set()
    bases_set = set(int(x) for x in bases_set) if bases_set is not None else set()

    # 1) 提取 new_route 中的 allowed customer 顺序（只看卡车路线里的 customer）
    new_seq = []
    for x in new_route:
        try:
            x = int(x)
        except Exception:
            continue
        if x == start_idx or x == depot_idx:
            continue
        if x in bases_set:
            continue
        if x in allowed_set and str(data.nodes[x].get("node_type", "")).lower() == "customer":
            if x not in new_seq:
                new_seq.append(x)

    # 2) 按 template 逐个替换（允许长度不一致）
    out = []
    k = 0
    for x in template_route:
        try:
            x = int(x)
        except Exception:
            continue

        if x == start_idx or x == depot_idx:
            continue

        if x in bases_set:
            out.append(x)
            continue

        is_customer = (0 <= x < len(data.nodes)) and (str(data.nodes[x].get("node_type", "")).lower() == "customer")
        if is_customer and (x in allowed_set):
            if k < len(new_seq):
                out.append(int(new_seq[k]))
                k += 1
            else:
                # 该客户在 new 解中已不再由卡车服务（应被无人机覆盖），从卡车模板里删除
                pass
        else:
            out.append(x)

    # 3) new_seq 还有剩余：说明 new 解新增了卡车客户（原本 template 没位置）=> 插入 depot 前
    if k < len(new_seq):
        exist = set(out)
        for x in new_seq[k:]:
            if x not in exist:
                out.append(int(x))
                exist.add(int(x))

    # 4) 保证首尾
    out = [start_idx] + [z for z in out if z != start_idx]
    if not out or out[-1] != depot_idx:
        out.append(depot_idx)

    return out

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
                node["reloc_type"] = str(predefined_types[int(c)])  # 中文注释：事件的相对可用时长（小时），用于后续在求解端计算 L 与有效截止时间
            if predefined_delta_avail is not None and int(c) in predefined_delta_avail:
                node["delta_avail_h"] = float(predefined_delta_avail[int(c)])
            else:
                node["delta_avail_h"] = float(node.get("delta_avail_h", 0.0))
            # 中文注释：候选有效截止时间 L（仅当最终接受且未服务时才生效）
            # 约定：ready_time/due_time 是“平台承诺窗”；effective_due（运行态）才表示“有效截止”
            node["candidate_effective_due"] = ut.cand_eff_due(node)

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
            if 2.0 * d <= sim.DRONE_RANGE_UNITS:
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

def quick_filter_relocations(data_cur, data_prelim, full_route_cur, full_b2d_cur, req_clients,
    decisions, alpha_drone, lambda_late, truck_speed, drone_speed, delta_cost_max=30.0,
    delta_late_max=0.10, prefix_route=None,):
    """    中文注释：对 apply_relocations_for_decision_time 的初判 decisions 做二次筛选（quick_filter）。

    在原 quick_filter 的基础上，按“锁死规则”加入：
    - 冻结承诺窗 READY_TIME/DUE_TIME；
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
            # 中文注释：卡车路径插入评估必须用 truck_arc_cost（包含路况系数等），否则会把插入位置选偏导致 Δtruck 虚高
            try:
                return float(sim.truck_arc_cost(data, int(i), int(j)))
            except Exception:
                # 兜底：如果 truck_arc_cost 不可用再退回 costMatrix
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
    late_prom_work = ut._total_late_against_due(
        data_work, route_work, b2d_work, res_work,
        due_mode="prom", drone_speed=drone_speed
    )

    qf_deltas = {}
    decisions_filtered = []

    # ---------- 3) 逐条叠加尝试 ACCEPT ----------
    for (cid, decision, nx, ny, reason) in decisions_norm:
        # 统一拿到承诺截止（PROM_DUE = due_time）
        prom_due = ut.prom_due(data_work.nodes[cid])

        # 若初判就不是 ACCEPT：按锁死规则回退 EFFECTIVE_DUE= PROM_DUE
        if decision != "ACCEPT":
            try:
                data_work.nodes[cid]["effective_due"] = prom_due
                # baseline 更新（后续请求以回退后的 due 继续评估）
                res_work = sim.evaluate_full_system(
                    data_work, route_work, b2d_work,
                    alpha_drone, lambda_late,
                    truck_speed, drone_speed
                )
                late_prom_work = ut._total_late_against_due(
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
            cand_due = ut.cand_eff_due(node_pre)
        else:
            try:
                cand_due = float(cand_due)
            except Exception:
                cand_due = ut.cand_eff_due(node_pre)

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

        # 2.5) 局部最小代价插入（cheapest-insertion）
        # 中文注释：
        # - 原来只对 truck→truck（old_in_route 且不强制卡车）做扫描；
        # - 但对 “无人机→force_truck=1” 这种兜底直送，如果只尾插会显著高估 Δtruck；
        # - 因此：只要 cid 当前在 route_trial 且不涉及 base_lock，就允许做一次后缀扫描（不破坏冻结前缀）。
        if (upd["base_lock"] is None) and (cid in route_trial):
            # 可选：debug 下比较尾插增量 vs 最优插入增量
            tail_delta = None
            if debug:
                try:
                    pos0 = route_trial.index(cid)
                    a0 = route_trial[pos0 - 1]
                    b0 = route_trial[pos0 + 1]
                    tail_delta = float(data_trial.costMatrix[int(a0)][int(cid)]) + \
                                 float(data_trial.costMatrix[int(cid)][int(b0)]) - \
                                 float(data_trial.costMatrix[int(a0)][int(b0)])
                except Exception:
                    tail_delta = None

            route_trial2, old_pos, best_pos, best_delta = _cheapest_reinsert_for_truck(
                data_trial, route_trial, cid, min_insert_pos, central_idx
            )
            route_trial = route_trial2

            if debug and (old_pos is not None) and (best_pos is not None) and (best_pos != old_pos):
                nid = data_cur.nodes[cid].get("node_id", cid)
                if tail_delta is not None:
                    print(f"[QF-INS] cid={cid} nid={nid} old_pos={old_pos} best_pos={best_pos} "
                          f"tailΔ={tail_delta:.3f} bestΔ={best_delta:.3f} improve={tail_delta - best_delta:.3f} "
                          f"min_pos={min_insert_pos}")
                else:
                    print(f"[QF-INS] cid={cid} nid={nid} old_pos={old_pos} best_pos={best_pos} "
                          f"bestΔ={best_delta:.3f} min_pos={min_insert_pos}")

        # 4) 评估 trial
        res1 = sim.evaluate_full_system(
            data_trial, route_trial, b2d_trial,
            alpha_drone, lambda_late,
            truck_speed, drone_speed
        )
        late_prom_trial = ut._total_late_against_due(
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
                res_work = sim.evaluate_full_system(
                    data_work, route_work, b2d_work,
                    alpha_drone, lambda_late,
                    truck_speed, drone_speed
                )
                late_prom_work = ut._total_late_against_due(
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

    return data_work, decisions_filtered, qf_deltas, route_work, b2d_work, res_work

def alns_truck_drone(data, base_to_drone_customers, max_iter=200, remove_fraction=None, T_start=None, T_end=None, alpha_drone=0.3, lambda_late=50.0, truck_customers=None, use_rl=False,
                     rl_tau=0.5, rl_eta=0.1, start_idx=None, start_time: float = 0.0, bases_to_visit=None,
                     ctx=None):
    if ctx is None:
        ctx = {}

        # 1) drone_range：自动兼容你工程里的常量名
        if "drone_range" not in ctx or ctx["drone_range"] is None:
            # 修改点：直接从 sim 模块获取
            ctx["drone_range"] = sim.DRONE_RANGE_UNITS

        if ctx["drone_range"] is None:
            raise RuntimeError("[GUARD] 未找到无人机航程常量：请检查 sim.DRONE_RANGE_UNITS")
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
    init_route = ctx.get("init_route", None)
    if init_route is not None and isinstance(init_route, (list, tuple)) and len(init_route) >= 2:
        # 中文注释：warm-start，用 quick_filter 的预定解作为 ALNS 初始解，避免“决策可行但 ALNS 重排后失控”
        current_route = list(init_route)
    else:
        # 原来的构造方式（例如 nearest neighbor）
        current_route = ops.nearest_neighbor_route_truck_only(
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
    init_cost = current_cost
    init_late = current_total_late


    # ===================== 收敛曲线追踪（A：best_cost_dist vs iteration） =====================
    # 中文注释：评价成本 cost_dist 不含迟到惩罚，仅用于画收敛曲线/对比，不改变 ALNS 求解逻辑。
    trace_enabled = bool(ctx.get("trace_converge", False))
    trace_csv_path = ctx.get("trace_csv_path", None)
    trace_rows = []

    def _flush_converge_trace():
        """中文注释：把 best-so-far 收敛轨迹写出到 CSV。
        说明：这是“记录/可视化”功能，不影响求解逻辑。
        """
        if (not trace_enabled) or (not trace_csv_path):
            return
        try:
            os.makedirs(os.path.dirname(str(trace_csv_path)) or '.', exist_ok=True)
            with open(trace_csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["iter", "best_cost_dist", "best_total_late", "best_truck_dist", "best_drone_dist"],
                )
                w.writeheader()
                for r in trace_rows:
                    w.writerow(r)
        except Exception as _e:
            # 中文注释：默认 verbose=False 不打印，避免污染终端；需要时可把 verbose=True 查看。
            if ctx.get('verbose', False):
                print('[WARN] converge trace save failed:', _e)
    # best_cost_dist_sofar：从迭代0起，记录“所有通过护栏的候选解”中 cost_dist 的最小值（best-so-far）。
    best_cost_dist_sofar = float(best_truck_dist + alpha_drone * best_drone_dist)
    best_late_for_bestdist = float(best_total_late)
    best_truck_dist_for_bestdist = float(best_truck_dist)
    best_drone_dist_for_bestdist = float(best_drone_dist)
    if trace_enabled:
        trace_rows.append({
            "iter": 0,
            "best_cost_dist": best_cost_dist_sofar,
            "best_total_late": best_late_for_bestdist,
            "best_truck_dist": float(best_truck_dist_for_bestdist),
                "best_drone_dist": float(best_drone_dist_for_bestdist),
            })
    if ctx.get("verbose", False):
        print(f"初始: 成本={current_cost:.3f}, 卡车距={current_truck_dist:.3f}, 无人机距={current_drone_dist:.3f}, 总迟到={current_total_late:.3f}")

    # === RL: 每个算子的得分（Q 值） ===
    n_inner = len(current_route) - 2
    if n_inner <= 0:
        # 兜底：仍需保证 force_truck 约束一致
        best_route, best_b2d = sim.enforce_force_truck_solution(data, best_route, best_b2d)

        # ===================== 收敛曲线写出（CSV） =====================
        _flush_converge_trace()

        return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time
    # ---------- ctx 处理：保留调用方传入的 dict 引用，用于回传算子统计 ----------
    ctx_in = ctx if isinstance(ctx, dict) else {}
    ctx = ctx_in
    ctx = ops.build_ab_cfg(ctx)

    DESTROYS = ctx.get("DESTROYS", [ops.D_random_route, ops.D_worst_route, ops.D_reloc_focus_v2, ops.D_switch_coverage])
    REPAIRS = ctx.get("REPAIRS",
                      [ops.R_greedy_only, ops.R_regret_only, ops.R_greedy_then_drone, ops.R_regret_then_drone,
                       ops.R_late_repair_reinsert, ops.R_base_feasible_drone_first])

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
        cand_route, cand_b2d = sim.enforce_force_truck_solution(data, cand_route, cand_b2d)


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

        # ---- 收敛曲线：在通过护栏的候选解上，更新 best_cost_dist_sofar（不依赖 SA 是否接受） ----
        if trace_enabled:
            cand_cost_dist = float(cand_truck_dist + alpha_drone * cand_drone_dist)
            if cand_cost_dist < best_cost_dist_sofar - 1e-12:
                best_cost_dist_sofar = cand_cost_dist
                best_late_for_bestdist = float(cand_total_late)
                best_truck_dist_for_bestdist = float(cand_truck_dist)
                best_drone_dist_for_bestdist = float(cand_drone_dist)
            trace_rows.append({
                "iter": int(it),
                "best_cost_dist": float(best_cost_dist_sofar),
                "best_total_late": float(best_late_for_bestdist),
                "best_truck_dist": float(best_truck_dist_for_bestdist),
                "best_drone_dist": float(best_drone_dist_for_bestdist),
            })

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
    best_route, best_b2d = sim.enforce_force_truck_solution(data, best_route, best_b2d)

    # ---------- 回传算子统计：用于消融验证（不影响求解逻辑） ----------
    try:
        ctx_in["__op_stat"] = op_stat
        ctx_in["__hit_stat"] = hit_stat
    except Exception:
        pass
    # ===== [POST-DBG] 极简诊断：确认 ALNS 最终返回的是 best（而不是 cur）=====
    if bool(ctx.get("dbg_postcheck", False)):
        print(f"[ALNS-DBG] init(cost={init_cost:.3f},late={init_late:.3f}) | "
              f"cur(cost={current_cost:.3f},late={current_total_late:.3f}) | "
              f"best(cost={best_cost:.3f},late={best_total_late:.3f})")

    # if bool(ctx.get("verbose_stat", True)):
    #     _print_operator_stats(op_stat, top_k=int(ctx.get("verbose_stat_topk", 30)))

    # ===================== 收敛曲线写出（CSV） =====================
    # 中文注释：正常迭代结束/提前结束（非 n_inner<=0）也需要写出；否则用户看不到 converge_*.csv。
    _flush_converge_trace()
    return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time


# dynamic_logic.py (追加在末尾)

def run_decision_epoch(
        decision_time,
        t_prev,
        scene_idx,
        data_cur,
        full_route_cur,
        full_b2d_cur,
        full_arrival_cur,
        full_depart_cur,
        full_finish_cur,
        offline_groups,
        nodeid2idx,
        ab_cfg,
        seed,
        verbose=False
):
    """
    执行单个决策时刻的完整流程：
    1. 切分时间轴 (Split)
    2. 应用事件与筛选 (Apply & Filter)
    3. 构造重规划集合 (Construct Set)
    4. ALNS 重规划 (Replan)
    5. 合并与全系统评估 (Merge & Evaluate)

    返回一个字典，包含更新后的状态、日志行、统计结果以及画图所需的数据包。
    """

    # 0. 随机种子同步
    if seed is not None:
        ut.set_seed(int(seed) + int(scene_idx))

    # 对照组/实验组标识：默认 G3（Full）
    method = str(ab_cfg.get("method", ab_cfg.get("compare_group", "G3"))).strip().upper()

    if verbose:
        print(f"\n===== 场景 {scene_idx}: 决策时刻 t = {decision_time:.2f} h 应用位置变更后 =====")

    # 1. Split: 切分已服务/未服务
    served_nodes, remaining_nodes, current_node, virtual_pos, prefix_route = split_route_by_decision_time(
        full_route_cur, full_arrival_cur, decision_time, data_cur.central_idx, data_cur
    )

    # 无人机已完成集合
    drone_served_set = set(get_drone_served_before_t(full_b2d_cur, full_finish_cur, decision_time))

    # 全系统已完成/未完成
    truck_served = set(served_nodes)
    served_all = truck_served | drone_served_set
    all_customers = {i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'customer'}
    unserved_all = all_customers - served_all

    # 2. 识别基站状态
    all_bases = [i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'base']
    if data_cur.central_idx not in all_bases:
        all_bases.append(data_cur.central_idx)
    route_set_cur = set(full_route_cur)
    bases_in_plan = [b for b in all_bases if (b in route_set_cur) or (b == data_cur.central_idx)]
    visited_bases = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) <= decision_time + 1e-9]
    bases_to_visit = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) > decision_time + 1e-9]
    feasible_bases_for_drone = sorted(set(visited_bases + bases_to_visit))
    # 动态阶段不允许从中心仓库起飞（车已离开）
    if decision_time > 1e-9:
        feasible_bases_for_drone = [b for b in feasible_bases_for_drone if b != data_cur.central_idx]
    arrival_prefix = {b: full_arrival_cur[b] for b in visited_bases if b in full_arrival_cur}
    # base_onhand：已访问基站上“尚未起飞”的无人机客户集合（用于公平对齐 visited-base 放飞规则）
    base_onhand = {int(b): set() for b in visited_bases}
    try:
        for _b, _cs in (full_b2d_cur or {}).items():
            _b = int(_b)
            if _b not in base_onhand:
                continue
            for _c in _cs:
                _c = int(_c)
                if _c in served_all:
                    continue
                if float(full_depart_cur.get(_c, float('inf'))) > decision_time + 1e-9:
                       base_onhand[_b].add(_c)
    except Exception:
         base_onhand = {int(b): set() for b in visited_bases}
    # 3. 处理离线事件 (Events)
    client_to_base_cur = build_client_to_base_map(full_b2d_cur)
    req_override = []
    predefined_xy = {}
    predefined_types = {}
    predefined_delta_avail = {}
    decision_log_rows = []

    key = round(float(decision_time), 6)
    evs = offline_groups.get(key, []) if offline_groups is not None else []
    _ev_meta = {}

    for e in evs:
        nid = int(e.get('NODE_ID', 0))
        cidx = nodeid2idx.get(nid, None)
        if cidx is None: continue

        # 记录过期事件
        if int(cidx) in served_all:
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
                'APPLIED_X': '', 'APPLIED_Y': '', 'FORCE_TRUCK': '', 'BASE_LOCK': ''
            })
            continue

        req_override.append(int(cidx))
        predefined_xy[int(cidx)] = (float(e.get('NEW_X', 0.0)), float(e.get('NEW_Y', 0.0)))
        predefined_delta_avail[int(cidx)] = float(e.get('DELTA_AVAIL_H', 0.0))
        predefined_types[int(cidx)] = ut.map_event_class_to_reloc_type(e.get('EVENT_CLASS', ''))
        _ev_meta[int(cidx)] = {
            'EVENT_ID': int(e.get('EVENT_ID', 0)),
            'EVENT_CLASS': str(e.get('EVENT_CLASS', '')),
            'DELTA_AVAIL_H': float(e.get('DELTA_AVAIL_H', 0.0))
        }

    # 去重
    seen = set()
    req_override = [c for c in req_override if (c not in seen and not seen.add(c))]
    # 应用变更 (Apply) / 快速筛选 (Quick Filter)
    data_prelim = data_cur
    decisions_raw = []
    req_clients = []

    # G0：不应用变更，全部拒绝（route/b2d/坐标均不更新）
    if method == "G0":
        data_next = data_cur
        decisions = []
        qf_deltas = {}
        qf_route_full, qf_b2d_full, qf_eval_preplan = None, None, None
        if len(req_override) > 0:
            for cid in req_override:
                cid = int(cid)
                nx, ny = predefined_xy.get(cid, (float(data_cur.nodes[cid].get('x', 0.0)), float(data_cur.nodes[cid].get('y', 0.0))))
                decisions.append((cid, "REJECT", float(nx), float(ny), "G0 No-Replan：全部拒绝请求，且不更新路线/坐标"))
        # req_clients 保持为空：后续不会进入 quick_filter/ALNS
        req_clients = []
    else:
        if len(req_override) > 0:
            data_cur._offline_ev_meta = _ev_meta
            data_prelim, decisions_raw, req_clients = apply_relocations_for_decision_time(
                data_cur, t_prev, decision_time,
                full_depart_cur, full_finish_cur, full_arrival_cur,
                client_to_base_cur,
                req_override, predefined_xy, predefined_types, predefined_delta_avail
            )
            try:
                data_cur._offline_ev_meta = {}
            except Exception:
                pass

    # 4. 判断是否提前结束 (Early Stop)
    unfinished_drone_exist = any(
        (full_finish_cur.get(_c, float('inf')) > decision_time + 1e-9) for _c in list(full_finish_cur.keys())
    )
    if (len(unserved_all) == 0) and (not unfinished_drone_exist) and (len(req_clients) == 0):
        if verbose:
            print(f"    [EARLY-STOP] t={decision_time:.2f}h：无未服务客户、无未完成无人机任务、且无新请求。")
        return {'break': True}

    # 5. 快速筛选 (Quick Filter) & 构造最终 Decisions（仅对非 G0 启用）
    if method != "G0":
        if len(req_clients) == 0:
            # 无新请求，直接沿用
            data_next = data_cur
            decisions = []
            qf_deltas = {}
            qf_route_full, qf_b2d_full, qf_eval_preplan = None, None, None
        else:
            # 中文注释：从 ab_cfg 读取 quick_filter / ALNS 关键参数；未配置则使用默认值。
            qf_cost_max = float(ab_cfg.get("qf_cost_max", ab_cfg.get("delta_cost_max", 30.0)))
            qf_late_max = float(ab_cfg.get("qf_late_max", ab_cfg.get("delta_late_max", 0.10)))

            remove_fraction_cfg = float(ab_cfg.get("remove_fraction", 0.10))
            sa_T_start = float(ab_cfg.get("sa_T_start", 50.0))
            sa_T_end = float(ab_cfg.get("sa_T_end", 1.0))
            min_remove_cfg = int(ab_cfg.get("min_remove", 3))

            # 中文注释：保底参数合法性（避免出现 0/负数温度导致 SA 退化）
            if remove_fraction_cfg <= 0:
                remove_fraction_cfg = 0.10
            if sa_T_start <= 0:
                sa_T_start = 50.0
            if sa_T_end <= 0:
                sa_T_end = 1.0
            if sa_T_start < sa_T_end:
                sa_T_start, sa_T_end = sa_T_end, sa_T_start
            if min_remove_cfg < 1:
                min_remove_cfg = 1
            data_next, decisions, qf_deltas, qf_route_full, qf_b2d_full, qf_eval_preplan = quick_filter_relocations(
                data_cur=data_cur,
                data_prelim=data_prelim,
                full_route_cur=full_route_cur,
                full_b2d_cur=full_b2d_cur,
                prefix_route=prefix_route,
                req_clients=req_clients,
                decisions=decisions_raw,
                alpha_drone=0.3,
                lambda_late=50.0,
                truck_speed=sim.TRUCK_SPEED_UNITS,
                drone_speed=sim.DRONE_SPEED_UNITS,
                delta_cost_max=qf_cost_max,
                delta_late_max=qf_late_max,
            )
    else:
        # G0：data_next/decisions 已在上方构造
        qf_route_full, qf_b2d_full, qf_eval_preplan = None, None, None

    # 6. 记录 Decision Log
    for (cid, dec, nx, ny, reason) in decisions:
        cid = int(cid)
        meta = _ev_meta.get(cid, {})
        _ox = float(data_cur.nodes[cid].get('x', 0.0))
        _oy = float(data_cur.nodes[cid].get('y', 0.0))
        _apx = float(data_next.nodes[cid].get('x', _ox))
        _apy = float(data_next.nodes[cid].get('y', _oy))
        drec = qf_deltas.get(cid, {})

        decision_log_rows.append({
            'EVENT_ID': int(meta.get('EVENT_ID', -1)),
            'EVENT_TIME': float(decision_time),
            'NODE_ID': int(data_cur.nodes[cid].get('node_id', cid)),
            'DECISION': str(dec),
            'REASON': str(reason),
            'OLD_X': _ox, 'OLD_Y': _oy,
            'NEW_X': float(nx), 'NEW_Y': float(ny),
            'EVENT_CLASS': str(meta.get('EVENT_CLASS', '')),
            'APPLIED_X': _apx, 'APPLIED_Y': _apy,
            'FORCE_TRUCK': int(data_next.nodes[cid].get('force_truck', 0)),
            'BASE_LOCK': (
                '' if data_next.nodes[cid].get('base_lock') is None else int(data_next.nodes[cid].get('base_lock'))),
            'DELTA_AVAIL_H': float(meta.get('DELTA_AVAIL_H', predefined_delta_avail.get(cid, 0.0))),
            'READY_TIME': float(data_next.nodes[cid].get('ready_time', 0.0)),
            'DUE_TIME': float(data_next.nodes[cid].get('due_time', 0.0)),
            'EFFECTIVE_DUE': float(data_next.nodes[cid].get('effective_due', data_next.nodes[cid].get('due_time', 0.0))),
            'D_COST': float(drec.get('D_COST', 0.0)),
            'D_LATE_PROM': float(drec.get('D_LATE_PROM', 0.0)),
            'D_LATE_EFF': float(drec.get('D_LATE_EFF', 0.0)),
        })

    # 7. 打印决策摘要
    if verbose:
        print("本次决策的客户变更结果：")
        if not decisions:
            print("  无客户在该时刻提出变更请求，或全部不满足条件。")
        else:
            for it in decisions:
                # 兼容不同格式
                if len(it) == 5:
                    c, dec, nx, ny, reason = it
                    nid = data_cur.nodes[int(c)]['node_id']
                    print(f"  client={nid}, dec={dec}, new=({nx:.2f},{ny:.2f}), reason={reason}")

    # 8. 判断是否跳过重规划 (Skip Replan)
    forced_truck = {i for i in data_next.customer_indices if data_next.nodes[i].get('force_truck', 0) == 1}
    num_req = len(decisions)
    num_acc = sum(1 for d in decisions if str(d[1]).startswith("ACCEPT"))
    num_rej = sum(1 for d in decisions if str(d[1]).startswith("REJECT"))

    # ====== G0 / G1：不进入 ALNS（对照组）======
    if method == "G0":
        # 沿用旧解评估（route/b2d/坐标均不更新）
        full_eval_g0 = sim.evaluate_full_system(
            data_next, full_route_cur, full_b2d_cur,
            alpha_drone=0.3, lambda_late=50.0,
            truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
        )
        stat_record = ut._pack_scene_record(
            scene_idx, decision_time, full_eval_g0,
            num_req=num_req, num_acc=num_acc, num_rej=num_rej,
            alpha_drone=0.3, lambda_late=50.0
        )
        return {
            'break': False,
            'skip': True,
            'data_next': data_next,
            'full_route_next': full_route_cur,
            'full_b2d_next': full_b2d_cur,
            'full_arrival_next': full_arrival_cur,
            'full_depart_next': full_depart_cur,
            'full_finish_next': full_finish_cur,
            'stat_record': stat_record,
            'decision_log_rows': decision_log_rows,
            'full_eval': full_eval_g0,
            'viz_pack': {
                'data': data_next,
                'route': full_route_cur,
                'b2d': full_b2d_cur,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }

    if method == "G1":
        # 只使用 quick_filter 的局部修补结果（若无请求则沿用旧解），不跑 ALNS
        full_route_next = list(qf_route_full) if (qf_route_full is not None) else list(full_route_cur)
        full_b2d_next = {b: list(cs) for b, cs in qf_b2d_full.items()} if (qf_b2d_full is not None) else {b: cs.copy() for b, cs in full_b2d_cur.items()}

        # 兜底: 未覆盖客户强制插入卡车
        full_route_next = sim.cover_uncovered_by_truck_suffix(
            data_next, full_route_next, full_b2d_next, prefix_len=len(prefix_route), unserved_customers=unserved_all, verbose=verbose
        )

        # 全系统排程 + 评估
        full_arrival_next, full_total_time_next, _ = sim.compute_truck_schedule(
            data_next, full_route_next, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
        )
        full_depart_next, full_finish_next, _ = sim.compute_multi_drone_schedule(
            data_next, full_b2d_next, full_arrival_next,
            num_drones_per_base=sim.NUM_DRONES_PER_BASE, drone_speed=sim.DRONE_SPEED_UNITS
        )
        full_finish_all_next = dict(full_arrival_next)
        for _cid, _fin in full_finish_next.items():
            if str(data_next.nodes[int(_cid)].get("node_type")).lower() == "customer":
                full_finish_all_next[int(_cid)] = float(_fin)

        full_eval_g1 = sim.evaluate_full_system(
            data_next, full_route_next, full_b2d_next,
            alpha_drone=0.3, lambda_late=50.0,
            truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
        )
        stat_record = ut._pack_scene_record(
            scene_idx, decision_time, full_eval_g1,
            num_req=num_req, num_acc=num_acc, num_rej=num_rej,
            alpha_drone=0.3, lambda_late=50.0
        )
        return {
            'break': False,
            'skip': True,
            'data_next': data_next,
            'full_route_next': full_route_next,
            'full_b2d_next': full_b2d_next,
            'full_arrival_next': full_arrival_next,
            'full_depart_next': full_depart_next,
            'full_finish_next': full_finish_all_next,
            'stat_record': stat_record,
            'decision_log_rows': decision_log_rows,
            'full_eval': full_eval_g1,
            'viz_pack': {
                'data': data_next,
                'route': full_route_next,
                'b2d': full_b2d_next,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }


    # 强制跳过重规划的条件：无ACCEPT 且 无force_truck变更
    if (num_acc == 0) and (len(forced_truck) == 0) and (len(req_clients) == 0):
        if verbose:
            print(f"    [SKIP-REPLAN] t={decision_time:.2f}h：无有效请求，跳过路径重规划。")

        # 沿用旧解评估
        full_eval_skip = sim.evaluate_full_system(
            data_next, full_route_cur, full_b2d_cur,
            alpha_drone=0.3, lambda_late=50.0,
            truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
        )

        stat_record = ut._pack_scene_record(
            scene_idx, decision_time, full_eval_skip,
            num_req=num_req, num_acc=num_acc, num_rej=num_rej,
            alpha_drone=0.3, lambda_late=50.0
        )

        return {
            'break': False,
            'skip': True,
            'data_next': data_next,
            'full_route_next': full_route_cur,
            'full_b2d_next': full_b2d_cur,
            'full_arrival_next': full_arrival_cur,
            'full_depart_next': full_depart_cur,
            'full_finish_next': full_finish_cur,
            'stat_record': stat_record,
            'decision_log_rows': decision_log_rows,
            'full_eval': full_eval_skip,
            # 画图需要的数据包
            'viz_pack': {
                'data': data_next,
                'route': full_route_cur,
                'b2d': full_b2d_cur,
                'decisions': decisions,
                'virtual_pos': virtual_pos,
                'prefix_route': prefix_route
            }
        }

    # 9. 准备 ALNS 重规划
    # 9.1 起点修正
    if virtual_pos is not None:
        start_idx_for_alns = add_virtual_truck_position_node(data_next, virtual_pos)
    else:
        start_idx_for_alns = current_node
        # 兜底：如果正好卡在 central 且已经走了，修正虚拟位置 (逻辑同主文件)
        if (start_idx_for_alns == data_next.central_idx and decision_time > 1e-6 and len(remaining_nodes) > 0):
            # (这里简化处理，直接用 current_node 也没大问题，保持与主文件逻辑一致即可)
            pass

    # 9.2 Allowed Customers
    req_all = {int(c) for c in req_clients}
    req_set = {c for c in req_all if c in unserved_all}
    unfinished_drone = set()
    for b, clients in full_b2d_cur.items():
        t_base = full_arrival_cur.get(b, float('inf'))
        for c in clients:
            if full_finish_cur.get(c, float('inf')) <= decision_time + 1e-9: continue
            if t_base > decision_time + 1e-9 or full_depart_cur.get(c, float('inf')) > decision_time + 1e-9:
                unfinished_drone.add(c)

    forced_truck_eff = forced_truck - truck_served  # 已服务的剔除
    allowed_customers = (set(remaining_nodes) | unfinished_drone | req_set | forced_truck_eff) - served_all

    # 9.3 构造结构集合 (Structure Sets)
    C_moved_accept = {int(it[0]) for it in decisions if str(it[1]).startswith("ACCEPT") and int(it[0]) in unserved_all}
    C_moved_reject = {int(it[0]) for it in decisions if str(it[1]).startswith("REJECT") and int(it[0]) in unserved_all}
    C_force_truck = {i for i in data_next.customer_indices if data_next.nodes[i].get("force_truck", 0) == 1}
    # Boundary (Top K)
    cand_boundary = [i for i in allowed_customers if i not in C_force_truck]

    def _boundary_score(i):
        xi, yi = data_next.nodes[i]["x"], data_next.nodes[i]["y"]
        best = float("inf")
        for b in feasible_bases_for_drone:
            xb, yb = data_next.nodes[b]["x"], data_next.nodes[b]["y"]
            d = ((xi - xb) ** 2 + (yi - yb) ** 2) ** 0.5
            best = min(best, abs(2 * d - sim.DRONE_RANGE_UNITS))
        return best

    cand_boundary.sort(key=_boundary_score)
    C_boundary = set(cand_boundary[:6])

    # 9.4 初始解分类 (Classify / Warm-start)
    use_preplan = (qf_route_full is not None) and (qf_b2d_full is not None) and (qf_eval_preplan is not None)

    def _extract_suffix_from_full(full_route, prefix_route, start_idx, data):
        # 中文注释：从“包含前缀”的 full_route 中取出本轮要优化的 suffix，并用 start_idx 作为起点
        if not full_route:
            return [start_idx, data.central_idx]
        last_fixed = prefix_route[-1] if (prefix_route is not None and len(prefix_route) > 0) else full_route[0]
        try:
            pos = full_route.index(last_fixed)
        except Exception:
            pos = 0
        tail = full_route[pos + 1:]  # last_fixed 后面的未来节点序列
        r = [start_idx] + tail
        if r[-1] != data.central_idx:
            r.append(data.central_idx)
        # 去掉相邻重复点（稳健）
        rr = []
        for x in r:
            if len(rr) == 0 or rr[-1] != x:
                rr.append(x)
        return rr

    if use_preplan:
        # 1) init_route：用于 ALNS 的初始卡车路线（含 base 节点/客户节点/central）
        init_route = _extract_suffix_from_full(qf_route_full, prefix_route, start_idx_for_alns, data_next)

        # 2) base_to_drone_next：直接继承 preplan 在 allowed_customers 范围内的无人机分配
        base_to_drone_next = {}
        drone_set = set()
        for b, cs in qf_b2d_full.items():
            cs2 = [c for c in cs if c in allowed_customers]
            if cs2:
                base_to_drone_next[b] = cs2
                drone_set |= set(cs2)

        # 3) truck_next：allowed_customers 中不在 drone_set 的客户；顺序尽量按 init_route 出现顺序
        truck_set = set(int(c) for c in allowed_customers) - drone_set
        truck_next = []
        for x in init_route:
            if x in truck_set and str(data_next.nodes[x].get("node_type", "")).lower() == "customer":
                truck_next.append(int(x))
        # 兜底：没出现在 init_route 里的也补上
        for c in truck_set:
            if c not in truck_next:
                truck_next.append(int(c))

    else:
        base_to_drone_next, truck_next = sim.classify_clients_for_drone(
            data_next, allowed_customers=allowed_customers, feasible_bases=feasible_bases_for_drone
        )
        init_route = None
    # =======================
    # 后缀口径：preplan 作为 ALNS 的基线（用于 POST-CHECK）
    # 中文注释：ALNS 内部 best_cost/best_late 是“从 decision_time 开始”的后缀目标；
    # 为了口径一致，这里也把 preplan 的后缀目标算出来。
    # =======================
    pre_suffix_cost = None
    pre_suffix_late = None
    alpha = float(ab_cfg.get("alpha_drone", 0.3))
    lam = float(ab_cfg.get("lambda_late", 50.0))
    if use_preplan and (init_route is not None):
        (pre_suffix_cost,
         _pre_td,
         _pre_dd,
         _pre_tlate,
         _pre_dlate,
         pre_suffix_late,
         _pre_ttime) = sim.evaluate_truck_drone_with_time(
            data_next,
            list(init_route),
            {b: list(cs) for b, cs in (base_to_drone_next or {}).items()},
            start_time=decision_time,
            arrival_prefix=arrival_prefix,
            alpha_drone=alpha,
            lambda_late=lam,
            truck_speed=sim.TRUCK_SPEED_UNITS,
            drone_speed=sim.DRONE_SPEED_UNITS
        )
    planner = str(ab_cfg.get("planner", "ALNS")).upper()
    # =======================
    # 调试：打印 ALNS/GRB 子问题输入集合（可行域）用于对齐核对
    # 开关：ab_cfg["dbg_planner_sets"]=True
    # =======================
    if bool(ab_cfg.get("dbg_planner_sets", False)):
        def _fmt_set(name, s, k=12):
            try:
                arr = sorted(int(x) for x in set(s))
            except Exception:
                arr = sorted(list(s))
            n = len(arr)
            if n <= k:
                show = arr
            else:
                h = k // 2
                show = arr[:h] + ["..."] + arr[-h:]
            return f"{name}: n={n}, sample={show}"

        print(f"\n[PLANNER-SETS][BEFORE] scene={scene_idx} t={decision_time:.2f} planner={planner} start_idx={start_idx_for_alns}")
        print(_fmt_set("served_all", served_all))
        print(_fmt_set("unserved_all", unserved_all))
        print(_fmt_set("visited_bases", visited_bases))
        print(_fmt_set("bases_to_visit", bases_to_visit))
        print(_fmt_set("feasible_bases_for_drone", feasible_bases_for_drone))
        print(_fmt_set("allowed_customers", allowed_customers))
        # 下面几类是你构造 allowed_customers 的来源，便于排查漏/多
        try:
            print(_fmt_set("req_set", req_set))
        except Exception:
            pass
        try:
            print(_fmt_set("unfinished_drone", unfinished_drone))
        except Exception:
            pass
        try:
            print(_fmt_set("forced_truck_eff", forced_truck_eff))
        except Exception:
            pass

    # 10. ALNS 求解 (Solve)
    ctx_for_alns = dict(ab_cfg)
    ctx_for_alns.update({
        "verbose": verbose,
        "C_moved_accept": C_moved_accept,
        "C_moved_reject": C_moved_reject,
        "C_force_truck": C_force_truck,
        "C_boundary": C_boundary,
        "min_remove": min_remove_cfg,
        "feasible_bases_for_drone": feasible_bases_for_drone,
        "visited_bases": set(visited_bases),"base_onhand": base_onhand,
        "arrival_prefix": arrival_prefix, "init_route": init_route,
        "dbg_postcheck": bool(ab_cfg.get("dbg_postcheck", False)),
    })
    # ===================== 收敛曲线开关/路径（仅对 ALNS 组生效） =====================
    # 默认关闭；在 ab_cfg 里设置 trace_converge=True 才会写出 converge_*.csv
    trace_on = bool(ab_cfg.get("trace_converge", False))
    if trace_on and (method in ("G2", "G3")):
        trace_dir = str(ab_cfg.get("trace_dir", "outputs"))
        try:
            os.makedirs(trace_dir, exist_ok=True)
        except Exception:
            pass
        ctx_for_alns["trace_converge"] = True
        ctx_for_alns["trace_csv_path"] = os.path.join(trace_dir, f"converge_{method}_seed{seed}_scene{scene_idx}.csv")
    else:
        ctx_for_alns["trace_converge"] = False
        ctx_for_alns["trace_csv_path"] = None

    if planner in ("GRB", "GUROBI"):
        # -----------------------
        # 走 Gurobi：滚动重规划后缀
        # -----------------------
        # 1) 组装子问题 DataFrame（只包含：start + depot + bases_to_visit + allowed_customers）
        rows = []
        seen = set()

        def _add_row(nid: int, ntype: str, due_override=None):
            if nid in seen:
                return
            nd = data_next.nodes[nid]
            due_val = due_override
            if due_val is None:
                due_val = float(nd.get("effective_due", nd.get("due_time", 0.0)))
            rows.append(dict(
                NODE_ID=int(nid),
                NODE_TYPE=str(ntype),
                ORIG_X=float(nd.get("x", 0.0)),
                ORIG_Y=float(nd.get("y", 0.0)),
                DEMAND=float(nd.get("demand", 0.0)),
                READY_TIME=float(nd.get("ready_time", 0.0)),
                DUE_TIME=float(due_val),
            ))
            seen.add(nid)

        visited_drone_only_bases = []
        try:
            for _b in visited_bases:
                _b = int(_b)
                if _b == int(data_next.central_idx):
                    continue
                if len(base_onhand.get(_b, set())) > 0:
                    visited_drone_only_bases.append(_b)
        except Exception:
            visited_drone_only_bases = []
        for b in visited_drone_only_bases:
            _add_row(int(b), "base", due_override=1e9)
        # start（虚拟车位置节点）——注意：类型用 truck_pos，避免被当 customer/base
        _add_row(start_idx_for_alns, "truck_pos", due_override=1e9)

        # depot（central）
        _add_row(data_next.central_idx, "central", due_override=1e9)

        # bases_to_visit（未来基站）
        for b in bases_to_visit:
            _add_row(int(b), "base", due_override=1e9)

        # customers（用 EFFECTIVE_DUE 优先作为 DUE）
        for c in allowed_customers:
            if str(data_next.nodes[c].get("node_type", "")).lower() != "customer":
                continue
            due_eff = data_next.nodes[c].get("effective_due", None)
            if due_eff is None:
                due_eff = data_next.nodes[c].get("due_time", 0.0)
            _add_row(int(c), "customer", due_override=float(due_eff))

        df_sub = pd.DataFrame(rows)

        # 2) 只允许从“未来基站”起飞（动态下通常不允许 depot 作为起飞点）
        allowed_bases = set(int(b) for b in bases_to_visit)
        for b in (visited_drone_only_bases or []):
            allowed_bases.add(int(b))
        drone_only_onhand = {int(b): sorted(int(c) for c in base_onhand.get(int(b), set())) for b in
                                 (visited_drone_only_bases or [])}
        # 3) 强制卡车客户（force_truck=1）
        ft = set()
        for c in allowed_customers:
            if int(data_next.nodes[c].get("force_truck", 0)) == 1:
                ft.add(int(c))
        must_visit_bases = set(int(b) for b in bases_to_visit)

        # 4) 调 Gurobi（每决策点 TimeLimit）
        res = grb.solve_milp_return_from_df(
            df_sub,
            unit_per_km=float(ab_cfg.get("unit_per_km", 5.0)),
            E_roundtrip_km=float(ab_cfg.get("E_roundtrip_km", 10.0)),
            truck_speed_kmh=float(ab_cfg.get("truck_speed_kmh", 30.0)),
            truck_road_factor=float(ab_cfg.get("truck_road_factor", 1.5)),
            drone_speed_kmh=float(ab_cfg.get("drone_speed_kmh", 60.0)),
            alpha=float(ab_cfg.get("alpha_drone", 0.3)),
            lambda_late=float(ab_cfg.get("lambda_late", 50.0)),
            time_limit=float(ab_cfg.get("grb_time_limit", 30.0)),
            mip_gap=float(ab_cfg.get("grb_mip_gap", 0.05)),
            allowed_bases=allowed_bases,
            must_visit_bases=must_visit_bases,
            drone_only_bases=set(visited_drone_only_bases or []),
            drone_only_onhand=drone_only_onhand,
            allow_depot_as_base=False,  # 动态后缀下建议关掉
            force_truck_customers=ft,
            verbose=int(ab_cfg.get("grb_verbose", 0)),
            start_node=int(start_idx_for_alns),
            start_time_h=float(decision_time),
        )

        if bool(ab_cfg.get("dbg_planner_sets", True)):
            print(f"[PLANNER-SETS][AFTER-GRB] sol_count={int(res.get('sol_count', 0))} gap={res.get('mip_gap', None)} obj={res.get('obj', None)}")
            # GRB 的输入集合
            print(f"[PLANNER-SETS][AFTER-GRB] df_sub_rows={len(df_sub)} allowed_bases(n)={len(allowed_bases)} force_truck(n)={len(ft)}")
            # GRB 的输出解结构
            try:
                r = res.get("route", [])
                da = res.get("drone_assign", {}) or {}
                print(f"[PLANNER-SETS][AFTER-GRB] route_len={len(r)} route_head={r[:10]} route_tail={r[-10:] if len(r)>10 else r}")
                print(f"[PLANNER-SETS][AFTER-GRB] drone_assign_bases={sorted(list(da.keys()))} total_drone_customers={sum(len(v) for v in da.values())}")
            except Exception as e:
                print("[PLANNER-SETS][AFTER-GRB] parse error:", e)

        alpha = float(ab_cfg.get("alpha_drone", 0.3))
        lam = float(ab_cfg.get("lambda_late", 50.0))
        if int(res.get("sol_count", 0)) <= 0 or (not res.get("route", [])):
            # 失败：回滚到 preplan 后缀
            route_next = list(init_route) if init_route is not None else []
            b2d_next = {b: list(cs) for b, cs in (base_to_drone_next or {}).items()}
            alns_suffix_cost = pre_suffix_cost
            alns_suffix_late = pre_suffix_late
        else:
            route_next = list(res["route"])
            b2d_next = {int(b): [int(c) for c in cs] for b, cs in (res.get("drone_assign") or {}).items()}
            # 1) 先保留 MILP 自己的 km 口径 obj（只用于打印/记录，不参与 post-check 比较）
            grb_obj_km = float(res.get("obj", 0.0))
            grb_late_h = float(res.get("late_sum", 0.0))

            # 2) 用你系统内部统一口径（坐标单位）重新评估这个 GRB 后缀解，保证与 pre_suffix_cost 同口径
            (alns_suffix_cost,
             _grb_td,
             _grb_dd,
             _grb_tlate,
             _grb_dlate,
             alns_suffix_late,
             _grb_ttime) = sim.evaluate_truck_drone_with_time(
                data_next,
                list(route_next),
                {b: list(cs) for b, cs in (b2d_next or {}).items()},
                start_time=decision_time,
                arrival_prefix=arrival_prefix,
                alpha_drone=alpha,
                lambda_late=lam,
                truck_speed=sim.TRUCK_SPEED_UNITS,
                drone_speed=sim.DRONE_SPEED_UNITS
            )

            # 3) 可选：打印两套口径，方便你确认单位差异（建议只在 grb_verbose 时打印）
            if int(ab_cfg.get("grb_verbose", 0)) >= 1:
                print(
                    f"[GRB] milp_obj_km={grb_obj_km:.3f} | suffix_cost_units={alns_suffix_cost:.3f} | late_h={alns_suffix_late:.3f}")
    else:
        # -----------------------
        # 走 ALNS（你原代码不动）
        # -----------------------
        (route_next, b2d_next, alns_suffix_cost, _, _, alns_suffix_late, _) = alns_truck_drone(
            data_next, base_to_drone_next, max_iter=1000,
            remove_fraction=remove_fraction_cfg, T_start=sa_T_start, T_end=sa_T_end, alpha_drone=0.3, lambda_late=50.0,
            truck_customers=truck_next, use_rl=False,
            start_idx=start_idx_for_alns, start_time=decision_time, bases_to_visit=bases_to_visit,
            ctx=ctx_for_alns
        )
        if bool(ab_cfg.get("dbg_planner_sets", True)):
            print(f"[PLANNER-SETS][AFTER-ALNS] suffix_cost={alns_suffix_cost:.3f} suffix_late={alns_suffix_late:.3f}")
            print(f"[PLANNER-SETS][AFTER-ALNS] route_len={len(route_next)} route_head={route_next[:10]} route_tail={route_next[-10:] if len(route_next)>10 else route_next}")
            try:
                total_d = sum(len(v) for v in (b2d_next or {}).values())
                print(f"[PLANNER-SETS][AFTER-ALNS] drone_assign_bases={sorted(list((b2d_next or {}).keys()))} total_drone_customers={total_d}")
            except Exception as e:
                print("[PLANNER-SETS][AFTER-ALNS] parse error:", e)

    # 11. 合并解 (Merge)

    # 中文注释：
    # full 评估必须在“同一条物理轨迹”上进行：
    # - suffix 求解的起点是 start_idx_for_alns（truck_pos 虚拟节点）
    # - 同时 use_preplan 时，冻结客户（allowed 之外的未服务客户）必须保留在后缀里，不能丢给 cover_uncovered 去乱插

    allowed_set_local = set(allowed_customers)

    # 1) 选后缀骨架：只有在“确实存在冻结客户”时才套用 preplan 的模板
    frozen_set = set(unserved_all) - set(allowed_customers)
    need_template_freeze = (
            use_preplan
            and (init_route is not None)
            and isinstance(init_route, (list, tuple))
            and (len(init_route) >= 2)
            and (len(frozen_set) > 0)
    )

    if need_template_freeze:
        suffix_template = list(init_route)
        suffix_for_full = _rewrite_allowed_order_on_template(
            data_next,
            template_route=suffix_template,
            allowed_set=allowed_set_local,
            new_route=list(route_next),
            start_idx=start_idx_for_alns,  # ✅ 必须补上这一行
            depot_idx=data_next.central_idx,
            bases_set=set(int(b) for b in bases_to_visit)
        )
    else:
        suffix_for_full = list(route_next)

    # 2) 稳健：确保 full 后缀以 truck_pos(start_idx) 开头（不要再删除它）
    if (not suffix_for_full) or (suffix_for_full[0] != start_idx_for_alns):
        suffix_for_full = [start_idx_for_alns] + [x for x in suffix_for_full if x != start_idx_for_alns]
        # 2.5) 若本轮为了保留冻结客户而套用了模板，则 suffix 的 post-check 也必须基于
        #      实际将要执行的 suffix_for_full（否则会出现“suffix 看起来更好，但 full 更差”的假象）
        if need_template_freeze:
            try:
                _c, _, _, _, _, _late, _ = sim.evaluate_truck_drone_with_time(
                    data_next,
                    suffix_for_full,
                    (b2d_next or {}),
                    start_time=float(decision_time),
                    arrival_prefix=arrival_prefix,
                    alpha_drone=0.3,
                    lambda_late=50.0,
                    truck_speed=sim.TRUCK_SPEED_UNITS,
                    drone_speed=sim.DRONE_SPEED_UNITS,
                    num_drones_per_base=sim.NUM_DRONES_PER_BASE,
                    default_base_arrival=float(decision_time),
                )
                alns_suffix_cost = float(_c)
                alns_suffix_late = float(_late)
            except Exception as e:
                if verbose or bool(ab_cfg.get("dbg_postcheck", False)):
                    print(f"[POST-DBG] recompute suffix_on_template failed: {e}")

    full_route_next = _merge_prefix_suffix(prefix_route, suffix_for_full)

    # 构造 full_b2d_next (保留历史 + 新增)
    full_b2d_next = {b: cs.copy() for b, cs in full_b2d_cur.items()}
    allowed_set = set(allowed_customers)
    # 移除 allowed 部分
    for b in full_b2d_next:
        full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in allowed_set]
    # 添加新分配
    for b, cs in b2d_next.items():
        full_b2d_next.setdefault(b, []).extend(cs)
    # 去重
    for b in full_b2d_next:
        full_b2d_next[b] = list(dict.fromkeys(full_b2d_next[b]))  # 保持顺序去重

    # 兜底: 未覆盖客户强制插入卡车
    full_route_next = sim.cover_uncovered_by_truck_suffix(
        data_next, full_route_next, full_b2d_next, prefix_len=len(prefix_route), unserved_customers=unserved_all, verbose=verbose
    )

    # 12. 全系统排程 (Full Schedule)
    full_arrival_next, full_total_time_next, _ = sim.compute_truck_schedule(
        data_next, full_route_next, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
    )
    full_depart_next, full_finish_next, _ = sim.compute_multi_drone_schedule(
        data_next, full_b2d_next, full_arrival_next,
        num_drones_per_base=sim.NUM_DRONES_PER_BASE, drone_speed=sim.DRONE_SPEED_UNITS
    )
    # 统一 Finish Time
    full_finish_all_next = dict(full_arrival_next)
    for _cid, _fin in full_finish_next.items():
        if str(data_next.nodes[int(_cid)].get("node_type")).lower() == "customer":
            full_finish_all_next[int(_cid)] = float(_fin)

    # 13. 评估 (Evaluate)
    full_eval = sim.evaluate_full_system(
        data_next, full_route_next, full_b2d_next,
        alpha_drone=0.3, lambda_late=50.0,
        truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
    )
    # 13.x 事后复核：若 ALNS 把 preplan 做坏了则回滚（保证“决策阈值”最终仍成立）
    if (qf_eval_preplan is not None) and (qf_route_full is not None) and (qf_b2d_full is not None):
        eps = float(ab_cfg.get("postcheck_eps", 1e-6))
        # 注意：下面字段名按你 evaluate_full_system 的实际 key 来改（你现在汇总表里有 cost(obj)、total_late）
        alns_cost = float(full_eval.get("cost(obj)", full_eval.get("cost", 0.0)))
        alns_late = float(full_eval.get("total_late", 0.0))
        pre_cost = float(qf_eval_preplan.get("cost(obj)", qf_eval_preplan.get("cost", 0.0)))
        pre_late = float(qf_eval_preplan.get("total_late", 0.0))

        # 中文注释：优先用“后缀口径”做回滚判断（与 ALNS 内部 best_cost/best_late 一致）
        # 若后缀口径不可用，则退回用 full_eval 与 preplan_full_eval 的全局口径比较。
        use_suffix_check = (
                (pre_suffix_cost is not None) and (pre_suffix_late is not None) and
                (alns_suffix_cost is not None) and (alns_suffix_late is not None)
        )

        # 1) suffix 口径是否变坏
        if use_suffix_check:
            bad_suffix = (alns_suffix_cost > pre_suffix_cost + eps) or (alns_suffix_late > pre_suffix_late + eps)
        else:
            bad_suffix = (alns_cost > pre_cost + eps) or (alns_late > pre_late + eps)

        # 2) full 口径是否违反“决策阈值”（保证最终仍满足 qf 阈值）
        qf_cost_max = float(ab_cfg.get("qf_cost_max", float("inf")))
        qf_late_max = float(ab_cfg.get("qf_late_max", float("inf")))

        bad_full = False
        if not math.isinf(qf_cost_max):
            if (alns_cost - pre_cost) > qf_cost_max + eps:
                bad_full = True
        if not math.isinf(qf_late_max):
            if (alns_late - pre_late) > qf_late_max + eps:
                bad_full = True

        bad = bad_suffix or bad_full

        # 中文注释：二次护栏——用“全局口径”的 late_hard / late_hard_delta 再兜底一次。
        # 解释：ALNS 内部 late_hard 约束的是“后缀口径”(决策点之后的剩余任务)；
        #       但拼回前缀后做 full_eval 时，总迟到可能会被放大（例如前缀/基站到达时刻变化、或评估口径差异）。
        late_abs_full = float(ab_cfg.get("late_hard_full", ab_cfg.get("late_hard", float("inf"))))
        late_delta_full = float(ab_cfg.get("late_hard_delta_full", ab_cfg.get("late_hard_delta", float("inf"))))
        full_late = float(full_eval.get("total_late", 0.0))
        pre_late_full = float(pre_late) if pre_late is not None else None

        bad_full_late = (full_late > late_abs_full + eps)
        if (pre_late_full is not None) and (not math.isinf(late_delta_full)):
            bad_full_late = bad_full_late or ((full_late - pre_late_full) > late_delta_full + eps)

        bad = bad or bad_full_late
        # ===== [POST-DBG] 极简诊断：只打印一行，确认用的是 suffix 还是 full =====
        if bool(ab_cfg.get("dbg_postcheck", False)):
            def _f(x):
                return "None" if x is None else f"{float(x):.3f}"
            planner_tag = str(ab_cfg.get("planner", "ALNS")).upper()
            if planner_tag in ("GRB", "GUROBI"):
                planner_tag = "GRB"
            elif planner_tag == "ALNS":
                planner_tag = "ALNS"
            mode = "SUFFIX" if use_suffix_check else "FULL"
            print(f"[POST-DBG] mode={mode} | "
                  f"suffix({planner_tag} cost={_f(alns_suffix_cost)}, late={_f(alns_suffix_late)} "
                  f"vs PRE cost={_f(pre_suffix_cost)}, late={_f(pre_suffix_late)}) | "
                  f"full({planner_tag} cost={_f(alns_cost)}, late={_f(alns_late)} "
                  f"vs PRE cost={_f(pre_cost)}, late={_f(pre_late)})")
        planner_tag = str(ab_cfg.get("planner", ab_cfg.get("planner_name", "ALNS"))).upper()
        disable_postcheck = (ab_cfg.get("disable_postcheck", 0) == 1)

        if (not disable_postcheck) and bad:
            # 1) 回滚必须无条件执行（不要放在 verbose/dbg 里面）
            #    下面这行用你自己的回滚函数/赋值替换
            # rollback_to_preplan() / route_next = pre_route_next / b2d_next = pre_b2d_next ...
            do_rollback = True

            # 2) 打印只是可选
            if verbose or bool(ab_cfg.get("dbg_postcheck", False)):
                reasons = []
                if bad_suffix:
                    reasons.append("suffix-worse")
                if bad_full:
                    reasons.append("qf-threshold")
                if bad_full_late:
                    reasons.append("late-hard")
                reason_tag = "+".join(reasons) if reasons else "unknown"

                label = "suffix" if (bad_suffix and not (bad_full or bad_full_late)) else "full"
                if label == "suffix":
                    print(f"[POST-CHECK] rollback ({label}:{reason_tag}): "
                          f"{planner_tag}(cost={alns_suffix_cost:.3f},late={alns_suffix_late:.3f}) vs "
                          f"PRE(cost={pre_suffix_cost:.3f},late={pre_suffix_late:.3f})")
                else:
                    print(f"[POST-CHECK] rollback ({label}:{reason_tag}): "
                          f"{planner_tag}(cost={alns_cost:.3f},late={alns_late:.3f}) vs "
                          f"PRE(cost={pre_cost:.3f},late={pre_late:.3f}) | "
                          f"Δcost={alns_cost - pre_cost:.3f} (max={qf_cost_max:.3f}) "
                          f"Δlate={alns_late - pre_late:.3f} (max={qf_late_max:.3f})")

            # 3) 真正执行回滚（示意；你用自己已有的那段“恢复 pre_*”代码替换）
            if do_rollback:
                # route_next = pre_route_next; b2d_next = pre_b2d_next; depart_next = pre_depart_next; ...
                pass

            full_route_next = list(qf_route_full)
            full_b2d_next = {b: list(cs) for b, cs in qf_b2d_full.items()}

            # 兜底覆盖 + 重算排程 + 重算评估（与下面原流程一致）
            full_route_next = sim.cover_uncovered_by_truck_suffix(
                data_next, full_route_next, full_b2d_next, prefix_len=len(prefix_route), unserved_customers=unserved_all, verbose=verbose
            )
            full_arrival_next, full_total_time_next, _ = sim.compute_truck_schedule(
                data_next, full_route_next, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
            )
            full_depart_next, full_finish_next, _ = sim.compute_multi_drone_schedule(
                data_next, full_b2d_next, full_arrival_next,
                num_drones_per_base=sim.NUM_DRONES_PER_BASE, drone_speed=sim.DRONE_SPEED_UNITS
            )
            full_finish_all_next = dict(full_arrival_next)
            for _cid, _fin in full_finish_next.items():
                if str(data_next.nodes[int(_cid)].get("node_type")).lower() == "customer":
                    full_finish_all_next[int(_cid)] = float(_fin)

            full_eval = sim.evaluate_full_system(
                data_next, full_route_next, full_b2d_next,
                alpha_drone=0.3, lambda_late=50.0,
                truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
            )

    stat_record = ut._pack_scene_record(
        scene_idx, decision_time, full_eval,
        num_req=num_req, num_acc=num_acc, num_rej=num_rej,
        alpha_drone=0.3, lambda_late=50.0
    )

    return {
        'break': False,
        'skip': False,
        'data_next': data_next,
        'full_route_next': full_route_next,
        'full_b2d_next': full_b2d_next,
        'full_arrival_next': full_arrival_next,
        'full_depart_next': full_depart_next,
        'full_finish_next': full_finish_all_next,
        'stat_record': stat_record,
        'decision_log_rows': decision_log_rows,
        'full_eval': full_eval,
        'viz_pack': {
            'data': data_next,
            'route': full_route_next,
            'b2d': full_b2d_next,
            'decisions': decisions,
            'virtual_pos': virtual_pos,
            'prefix_route': prefix_route
        }
    }