import copy
import math
import simulation as sim
import numpy as np
import utils as ut # apply函数依赖这个
from data_io_1322 import recompute_cost_and_nearest_base # 依赖这个重算距离

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
            # 中文注释：冻结承诺窗（若缺失则回退到现有 ready/due）
            if "prom_ready" not in node:
                node["prom_ready"] = float(node.get("ready_time", 0.0))
            if "prom_due" not in node:
                node["prom_due"] = float(node.get("due_time", node.get("prom_ready", 0.0)))
            # 中文注释：候选有效截止时间 L（仅当最终接受且未服务时才生效）
            node["candidate_effective_due"] = max(float(node["prom_due"]),
                                                  float(node["prom_ready"]) + float(node["delta_avail_h"]))
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
    late_prom_work = ut._total_late_against_due(
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
                data_work.nodes[cid]["due_time"] = prom_due
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

    return data_work, decisions_filtered, qf_deltas
