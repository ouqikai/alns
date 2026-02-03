# operators.py
import random
import simulation as sim

# =========================================================
# 工具函数：保护节点 / 可复现采样 / 统一读取 ctx 里的集合
# =========================================================

def _get_protected_nodes(ctx, data):
    """统一构造 protected_nodes。默认额外保护所有 base 节点，避免 destroy 误删基站。"""
    protected = set(ctx.get("protected_nodes", set()))
    if ctx.get("auto_protect_bases", True):
        try:
            for i, nd in enumerate(getattr(data, "nodes", [])):
                if isinstance(nd, dict) and nd.get("node_type") == "base":
                    protected.add(i)
        except Exception:
            pass
    return protected

def _as_set(ctx, *keys):
    """把 ctx 中多个候选 key 的集合合并成 set（不存在则忽略）。"""
    out = set()
    for k in keys:
        v = ctx.get(k, None)
        if v:
            out |= set(v)
    return out

def _sorted_if(lst, ctx):
    """当 deterministic_order=True 时，对列表排序，避免 set 遍历顺序导致同 seed 不稳定。"""
    if ctx.get("deterministic_order", False):
        try:
            return sorted(lst)
        except Exception:
            return list(lst)
    return list(lst)

def _sample(pop, k, ctx):
    """可复现采样：deterministic_order=True 时，先排序再 sample。"""
    pop2 = _sorted_if(pop, ctx)
    if k <= 0:
        return []
    if k >= len(pop2):
        return pop2[:]
    return random.sample(pop2, k)
# =========================================================
# 基础插入/移除原语 (Primitives)
# =========================================================

def random_removal(route, num_remove, data, protected=None):
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


def worst_removal(route, num_remove, data, protected=None):
    """
    删除对当前卡车距离贡献最大的若干节点
    """
    if protected is None:
        protected = set()
    # 内部节点位置（不含首尾）
    inner_positions = list(range(1, len(route) - 1))
    if not inner_positions:
        return route[:], []

    # 计算每个节点的“贡献”
    contributions = []
    for pos in inner_positions:
        i = route[pos]
        if i in protected:
            continue
        a = route[pos - 1]
        b = route[pos + 1]
        # 使用 sim 模块计算距离
        saving = (sim.truck_arc_cost(data, a, i) +
                  sim.truck_arc_cost(data, i, b) -
                  sim.truck_arc_cost(data, a, b))
        contributions.append((saving, pos, i))

    contributions.sort(reverse=True, key=lambda x: x[0])
    to_remove = [pos for (_, pos, _) in contributions[:num_remove]]

    to_remove_set = set(to_remove)
    destroyed_route = [node for idx, node in enumerate(route) if idx not in to_remove_set]
    removed_nodes = [route[pos] for pos in to_remove]

    return destroyed_route, removed_nodes


def greedy_insert(data, route, removed_nodes):
    """
    大邻域中的 '修复算子'：贪心插入
    """
    new_route = route[:]

    for node in removed_nodes:
        best_pos = None
        best_delta = float('inf')

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


def regret_insert(data, destroyed_route, removed_nodes):
    """
    Regret-2 插入
    """
    route = destroyed_route[:]

    if not removed_nodes:
        return route

    while removed_nodes:
        best_k = None
        best_pos = None
        best_delta = None
        best_regret = -1e9

        for k in removed_nodes:
            deltas = []
            for pos in range(1, len(route)):
                a = route[pos - 1]
                b = route[pos]
                delta = (sim.truck_arc_cost(data, a, k) +
                         sim.truck_arc_cost(data, k, b) -
                         sim.truck_arc_cost(data, a, b))
                deltas.append((delta, pos))

            deltas.sort(key=lambda x: x[0])
            best1_delta, best1_pos = deltas[0]
            if len(deltas) > 1:
                best2_delta = deltas[1][0]
            else:
                best2_delta = best1_delta

            regret = best2_delta - best1_delta

            if regret > best_regret:
                best_regret = regret
                best_k = k
                best_pos = best1_pos
                best_delta = best1_delta

        route.insert(best_pos, best_k)
        removed_nodes.remove(best_k)

    return route


# =========================================================
# 具体 Destroy 算子
# =========================================================

def D_random_route(data, route, b2d, ctx):
    """通用随机破坏：随机移除 route 内若干客户（默认不动基站）。"""
    num_remove = int(ctx.get("num_remove", 1))
    protected = _get_protected_nodes(ctx, data)
    destroyed_route, removed = random_removal(route, num_remove, data, protected=protected)
    destroyed_b2d = {b: lst[:] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def D_worst_route(data, route, b2d, ctx):
    """通用成本导向破坏：移除对卡车距离贡献最大的若干客户（默认不动基站）。"""
    num_remove = int(ctx.get("num_remove", 1))
    protected = _get_protected_nodes(ctx, data)
    destroyed_route, removed = worst_removal(route, num_remove, data, protected=protected)
    destroyed_b2d = {b: lst[:] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def D_reloc_focus_v2(data, route, b2d, ctx):
    """
    面向“客户位置变更”的聚焦破坏（建议作为主力 destroy）：
    - 优先拔“变更被拒绝 / 强制卡车 / 边界敏感”客户（更有利于提高接纳率）
    - 其次少量拔已接受变更客户（用于局部重排，避免被早期决策锁死）
    - 可选：当你同时启用 D_switch_coverage 时，可排除“明显可切换基站”的客户，降低功能重叠
    """
    num_remove = int(ctx.get("num_remove", 1))
    protected = _get_protected_nodes(ctx, data)

    # 统一读取集合（兼容旧 key）
    moved_acc = _as_set(ctx, "C_moved_accept")
    moved_rej = _as_set(ctx, "C_moved_reject")
    force_set = _as_set(ctx, "C_force_truck", "force_truck_set")
    boundary = _as_set(ctx, "C_boundary")

    # 默认策略：先处理“难点”（rej/force/boundary），再少量处理 acc
    mode = ctx.get("reloc_focus_mode", "rej_first")
    if mode == "orig":
        pool_main = (moved_acc | force_set | boundary)
        pool_soft = (moved_rej - pool_main)
    else:
        pool_main = (moved_rej | force_set | boundary)
        pool_soft = (moved_acc - pool_main)

    def is_removable(i):
        return (0 <= int(i) < len(data.nodes)
                and data.nodes[int(i)].get("node_type") == "customer"
                and int(i) not in protected)

    pool_main = [int(i) for i in pool_main if is_removable(i)]
    pool_soft = [int(i) for i in pool_soft if is_removable(i)]

    # 可选：降低与 D_switch_coverage 的重叠——把“明显可切换基站”的客户从 pool_soft 中排除
    if ctx.get("reloc_focus_exclude_switchable", True) and ctx.get("use_switch_coverage_destroy", True):
        bases_to_visit = ctx.get("bases_to_visit", [])
        drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)
        route_set = set(route)
        def _switchable(cid: int) -> bool:
            try:
                feas = sim.feasible_bases_for_customer(data, cid, ctx, route_set, drone_range)
                lockb = data.nodes[cid].get("base_lock", None)
                if lockb is not None:
                    feas = [b for b in feas if b == lockb]
                return len(feas) > 0
            except Exception:
                return False
        # 只排除 soft（主要是 moved_acc/boundary），避免把 moved_rej 的“难点”误排除
        pool_soft = [cid for cid in pool_soft if not _switchable(cid)]

    removed = []
    if pool_main:
        take = min(num_remove, len(pool_main))
        removed.extend(_sample(pool_main, take, ctx))

    # soft 只补一点点，避免过度重排已较稳定的部分
    if len(removed) < num_remove and pool_soft:
        need = num_remove - len(removed)
        cap = max(1, num_remove // 3)
        take = min(need, cap, len(pool_soft))
        removed.extend(_sample(pool_soft, take, ctx))

    # 邻居增强：把被拔客户附近的一个邻居也考虑进来，提升局部重排效果
    route_pos = {node: idx for idx, node in enumerate(route)}
    extra = []
    for c in list(removed):
        if c not in route_pos:
            continue
        j = route_pos[c]
        for nb in [route[j - 1] if j - 1 >= 0 else None, route[j + 1] if j + 1 < len(route) else None]:
            if nb is None:
                continue
            nb = int(nb)
            if is_removable(nb) and nb not in removed and nb not in extra:
                extra.append(nb)

    extra = _sorted_if(extra, ctx)
    random.shuffle(extra)  # 保留一定随机性
    for nb in extra:
        if len(removed) >= num_remove:
            break
        removed.append(nb)

    # 兜底：不够就从 route 内补
    if len(removed) < num_remove:
        inner = [int(i) for i in route[1:-1] if is_removable(i) and int(i) not in set(removed)]
        need = num_remove - len(removed)
        if inner:
            removed += _sample(inner, min(need, len(inner)), ctx)

    removed_set = set(removed)
    destroyed_route = [x for x in route if x not in removed_set]
    destroyed_b2d = {b: [c for c in lst if c not in removed_set] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def D_switch_coverage(data, route, b2d, ctx):
    """
    覆盖切换导向破坏（建议作为主力 destroy）：
    - 只拔“存在可行基站集合”的客户，给 repair 创造 truck↔drone / base-switch 的空间
    - 与 D_reloc_focus_v2 互补：它更偏“难点客户/边界”，本算子更偏“可切换客户”
    """
    num_remove = int(ctx.get("num_remove", 1))
    protected = _get_protected_nodes(ctx, data)

    route_set = set(route)
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)

    # 兼容两种 key
    force_set = _as_set(ctx, "force_truck_set", "C_force_truck")

    in_route = {int(i) for i in route if data.nodes[int(i)].get("node_type") == "customer"}
    in_drone = {int(c) for cs in b2d.values() for c in cs}
    cand = list((in_route | in_drone) - force_set)

    def ok(i: int) -> bool:
        if i in protected:
            return False
        if data.nodes[i].get("node_type") != "customer":
            return False
        try:
            feas = sim.feasible_bases_for_customer(data, i, ctx, route_set, drone_range)
            lockb = data.nodes[i].get("base_lock", None)
            if lockb is not None:
                feas = [b for b in feas if b == lockb]
            return len(feas) > 0
        except Exception:
            return False

    cand = [int(i) for i in cand if ok(int(i))]
    if not cand:
        return route[:], {b: lst[:] for b, lst in b2d.items()}, []

    removed = _sample(cand, min(num_remove, len(cand)), ctx)
    removed_set = set(removed)

    destroyed_route = [x for x in route if x not in removed_set]
    destroyed_b2d = {b: [c for c in lst if c not in removed_set] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def D_late_worst(data, route, b2d, ctx):
    """
    迟到驱动破坏（建议低频使用）：
    - 只在“总迟到超过阈值”时才真正按迟到从大到小拔点
    - 若迟到很小/几乎为 0，则自动退化为 D_worst_route（避免重复追迟到）
    """
    num_remove = int(ctx.get("num_remove", 1))
    protected = _get_protected_nodes(ctx, data)

    start_time = float(ctx.get("start_time", 0.0))
    truck_speed = float(ctx.get("truck_speed", sim.TRUCK_SPEED_UNITS))
    arrival_prefix = ctx.get("arrival_prefix", None)

    # 计算当前卡车路线到达时间
    arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
    if arrival_prefix:
        arrival_times = dict(arrival_times)
        arrival_times.update(arrival_prefix)

    # 扫描卡车客户迟到
    late_customers = []
    total_late = 0.0
    for idx in route:
        idx = int(idx)
        if idx in protected:
            continue
        if data.nodes[idx].get("node_type") == "customer":
            due = float(data.nodes[idx].get("effective_due", data.nodes[idx].get("due_time", float("inf"))))
            arr = float(arrival_times.get(idx, 0.0))
            late = max(0.0, arr - due)
            if late > 1e-9:
                total_late += late
                late_customers.append((late, idx))

    trigger = float(ctx.get("LATE_DESTROY_TRIGGER", 1e-6))
    if total_late <= trigger:
        # 迟到不严重时，退化为成本导向破坏
        return D_worst_route(data, route, b2d, ctx)

    late_customers.sort(key=lambda x: x[0], reverse=True)
    removed = [cid for _, cid in late_customers[:num_remove]]

    # 不够再补一点随机（保持多样性）
    if len(removed) < num_remove:
        remaining_inner = [int(x) for x in route[1:-1] if int(x) not in protected and int(x) not in set(removed)]
        need = num_remove - len(removed)
        removed.extend(_sample(remaining_inner, min(need, len(remaining_inner)), ctx))

    removed_set = set(removed)
    destroyed_route = [x for x in route if x not in removed_set]
    destroyed_b2d = {b: [c for c in lst if c not in removed_set] for b, lst in b2d.items()}
    return destroyed_route, destroyed_b2d, removed

def drone_repair_feasible(data, route, b2d, ctx, k_moves=5, sample_k=12):
    bases_to_visit = ctx.get("bases_to_visit", [])
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)
    alpha_drone = ctx.get("alpha_drone", 0.3)
    lambda_late = ctx.get("lambda_late", 50.0)
    truck_speed = ctx.get("truck_speed", sim.TRUCK_SPEED_UNITS)
    drone_speed = ctx.get("drone_speed", sim.DRONE_SPEED_UNITS)
    start_time = ctx.get("start_time", 0.0)

    route_set = set(route)
    force_truck_set = set(ctx.get("force_truck_set", set()))

    in_route = {i for i in route if data.nodes[i].get("node_type") == "customer"}
    in_drone = {c for cs in b2d.values() for c in cs}

    candidates = list((in_route | in_drone) - force_truck_set)

    def eval_cost(r, bd):
        # 调用 sim.evaluate
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
        best_sol = None

        for cid in pool:
            node = data.nodes[cid]
            locked_b = node.get("base_lock", None)

            # Move 1: truck -> drone
            if cid in in_route:
                feas_bases = sim.feasible_bases_for_customer(
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

            # Move 2: drone -> truck
            if cid in in_drone:
                bd2 = {bb: [c for c in lst if c != cid] for bb, lst in b2d.items()}
                r2 = greedy_insert(data, route, [cid])
                new_cost = eval_cost(r2, bd2)
                delta = new_cost - cur_cost
                if delta < best_delta:
                    best_delta = delta
                    best_sol = (r2, bd2, new_cost)

            # Move 3: drone switch base
            if cid in in_drone:
                cur_b = None
                for bb, lst in b2d.items():
                    if cid in lst:
                        cur_b = bb
                        break
                if cur_b is not None:
                    feas_bases = sim.feasible_bases_for_customer(
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


def R_greedy_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r = greedy_insert(data, destroyed_route, removed_customers)
    bd = {b: lst[:] for b, lst in destroyed_b2d.items()}

    return r, bd


def R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r = regret_insert(data, destroyed_route, removed_customers)
    bd = {b: lst[:] for b, lst in destroyed_b2d.items()}

    return r, bd


def R_greedy_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    """先 greedy 插回卡车，再做一次 drone/base 可行修复。"""
    r, bd = R_greedy_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    return r, bd

def R_regret_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    """主力 repair：先 regret 插回卡车，再做一次 drone/base 可行修复。"""
    r, bd = R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    return r, bd

def R_base_feasible_drone_first(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    """
    无人机优先修复（偏“提高接纳率”）：
    - removed 客户若存在可行基站，优先分配到无人机
    - 为了控成本/迟到：支持 base 选择策略
        ctx['drone_first_pick'] = 'random'（原行为，默认）
                            或 'min_dist'（选 drone 距离最短）
                            或 'min_obj'（粗估 alpha*dist + lambda*late，推荐）
    """
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)

    route = destroyed_route[:]
    b2d = {b: lst[:] for b, lst in destroyed_b2d.items()}
    route_set = set(route)
    force_set = _as_set(ctx, "force_truck_set", "C_force_truck")

    pick_mode = ctx.get("drone_first_pick", "random")
    start_time = float(ctx.get("start_time", 0.0))
    truck_speed = float(ctx.get("truck_speed", sim.TRUCK_SPEED_UNITS))
    drone_speed = float(ctx.get("drone_speed", sim.DRONE_SPEED_UNITS))
    alpha_drone = float(ctx.get("alpha_drone", 0.3))
    lambda_late = float(ctx.get("lambda_late", 50.0))
    arrival_prefix = ctx.get("arrival_prefix", None)

    # 预先算一次卡车到达时间（用于 min_obj）
    arrival_times = None
    if pick_mode == "min_obj":
        try:
            arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
            if arrival_prefix:
                arrival_times = dict(arrival_times)
                arrival_times.update(arrival_prefix)
        except Exception:
            arrival_times = None

    def _pick_best_base(cid: int, feas: list):
        if not feas:
            return None
        if pick_mode == "random":
            return random.choice(feas)
        if pick_mode == "min_dist":
            return min(feas, key=lambda b: float(data.costMatrix[int(b)][cid]))
        # min_obj：粗估 obj = alpha*dist + lambda*late（只看该客户的 late）
        if arrival_times is None:
            return min(feas, key=lambda b: float(data.costMatrix[int(b)][cid]))
        due = float(data.nodes[cid].get("effective_due", data.nodes[cid].get("due_time", float("inf"))))
        best_b = None
        best_val = float("inf")
        for b in feas:
            b = int(b)
            t_b = float(arrival_times.get(b, 0.0))
            d = float(data.costMatrix[b][cid])
            t_svc = t_b + d / max(1e-9, float(drone_speed))
            late = max(0.0, t_svc - due)
            val = alpha_drone * d + lambda_late * late
            if val < best_val:
                best_val = val
                best_b = b
        return best_b

    for cid in removed_customers:
        cid = int(cid)
        if cid in force_set:
            continue

        feas = sim.feasible_bases_for_customer(data, cid, ctx, route_set, drone_range)
        lockb = data.nodes[cid].get("base_lock", None)
        if lockb is not None:
            feas = [b for b in feas if int(b) == int(lockb)]

        if feas:
            b = _pick_best_base(cid, feas)
            if b is None:
                b = int(random.choice(feas))
            b2d.setdefault(int(b), [])
            b2d[int(b)].append(cid)
        else:
            # 兜底：插回卡车
            route = greedy_insert(data, route, [cid])
            route_set = set(route)
            # route 改了，min_obj 需要刷新 arrival_times
            if pick_mode == "min_obj":
                try:
                    arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
                    if arrival_prefix:
                        arrival_times = dict(arrival_times)
                        arrival_times.update(arrival_prefix)
                except Exception:
                    arrival_times = None

    return route, b2d

def _late_repair_score_bases_by_drone_lateness(
        data, route, base_to_drone_customers,
        start_time, truck_speed, drone_speed,
        arrival_prefix=None, eps=1e-9,
):
    if not base_to_drone_customers:
        return []

    arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
    if arrival_prefix:
        arrival_times = dict(arrival_times)
        arrival_times.update(arrival_prefix)

    scored = []
    for b, cs in base_to_drone_customers.items():
        b = int(b)
        if b not in arrival_times:
            continue
        t_b = float(arrival_times[b])
        sum_late = 0.0
        max_late = 0.0
        for c in cs:
            c = int(c)
            due = float(data.nodes[c].get('effective_due', data.nodes[c].get('due_time', float('inf'))))
            if not (due < float('inf')):
                continue
            d = float(data.costMatrix[b][c])
            t_svc = t_b + d / float(drone_speed)
            late = max(0.0, t_svc - due)
            if late > eps:
                sum_late += late
                if late > max_late:
                    max_late = late
        if sum_late > eps:
            scored.append((sum_late, max_late, b))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored


def _late_repair_best_reinsert_position(data, route, base_to_drone_customers, cid, ctx):
    if (route is None) or (len(route) < 3) or (cid not in route):
        return None

    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', sim.TRUCK_SPEED_UNITS))
    drone_speed = float(ctx.get('drone_speed', sim.DRONE_SPEED_UNITS))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 50.0))
    arrival_prefix = ctx.get('arrival_prefix', None)
    eps = float(ctx.get('eps', 1e-9))

    base_cost, _, _, _, _, base_late, _ = sim.evaluate_truck_drone_with_time(
        data, route, base_to_drone_customers,
        start_time, truck_speed, drone_speed,
        alpha_drone, lambda_late, arrival_prefix=arrival_prefix,
    )

    old_pos = int(route.index(cid))
    base_route = route[:]
    base_route.pop(old_pos)
    if len(base_route) < 2:
        return None

    best_cost = float(base_cost)
    best_late = float(base_late)
    best_pos = int(old_pos)
    best_route = route

    for pos in range(1, len(base_route)):
        trial = base_route[:]
        trial.insert(pos, cid)
        cost, _, _, _, _, total_late, _ = sim.evaluate_truck_drone_with_time(
            data, trial, base_to_drone_customers,
            start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late, arrival_prefix=arrival_prefix,
        )

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


def _late_repair_best_base_reinsert(data, route, base_to_drone_customers, base_idx, ctx):
    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', sim.TRUCK_SPEED_UNITS))
    drone_speed = float(ctx.get('drone_speed', sim.DRONE_SPEED_UNITS))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 50.0))
    arrival_prefix = ctx.get('arrival_prefix', None)
    eps = float(ctx.get('eps', 1e-9))

    try:
        base_cost, _, _, _, _, base_late, _ = sim.evaluate_truck_drone_with_time(
            data, route, base_to_drone_customers,
            start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late, arrival_prefix=arrival_prefix,
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

    for pos in range(1, len(route_wo)):
        if pos == old_pos:
            pass
        trial = route_wo[:pos] + [base_idx] + route_wo[pos:]
        try:
            cost, _, _, _, _, total_late, _ = sim.evaluate_truck_drone_with_time(
                data, trial, base_to_drone_customers,
                start_time, truck_speed, drone_speed,
                alpha_drone, lambda_late, arrival_prefix=arrival_prefix,
            )
        except Exception:
            continue

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


def late_repair_truck_reinsert(data, route, base_to_drone_customers, ctx):
    max_moves = int(ctx.get('LATE_REPAIR_MAX_MOVES', 3))
    eps = float(ctx.get('eps', 1e-9))
    start_time = float(ctx.get('start_time', 0.0))
    truck_speed = float(ctx.get('truck_speed', sim.TRUCK_SPEED_UNITS))
    drone_speed = float(ctx.get('drone_speed', sim.DRONE_SPEED_UNITS))
    alpha_drone = float(ctx.get('alpha_drone', 0.3))
    lambda_late = float(ctx.get('lambda_late', 50.0))
    arrival_prefix = ctx.get('arrival_prefix', None)

    def _eval(r):
        return sim.evaluate_truck_drone_with_time(
            data, r, base_to_drone_customers,
            start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late,
            arrival_prefix=arrival_prefix,
        )

    try:
        best_cost, _, _, _, _, best_late, _ = _eval(route)
    except Exception:
        return route

    for _ in range(max_moves):
        if best_late <= eps:
            break

        arrival_times, _, _ = sim.compute_truck_schedule(data, route, start_time, truck_speed)
        if arrival_prefix:
            arrival_times = dict(arrival_times)
            arrival_times.update(arrival_prefix)

        worst_truck = None
        for idx in route:
            nt = str(data.nodes[idx].get('node_type', '')).lower()
            if nt not in ('customer', 'truck_customer', 'truck'):
                continue
            due = float(data.nodes[idx].get('effective_due', data.nodes[idx].get('due_time', float('inf'))))
            if not (due < float('inf')):
                continue
            t = float(arrival_times.get(idx, 0.0))
            late = max(0.0, t - due)
            if late > eps and (worst_truck is None or late > worst_truck[0]):
                worst_truck = (late, int(idx))

        base_scores = _late_repair_score_bases_by_drone_lateness(
            data, route, base_to_drone_customers,
            start_time=start_time, truck_speed=truck_speed,
            drone_speed=drone_speed, arrival_prefix=arrival_prefix, eps=eps,
        )
        worst_base = base_scores[0] if base_scores else None

        truck_signal = worst_truck[0] if worst_truck else 0.0
        drone_signal = worst_base[0] if worst_base else 0.0

        improved = False

        if truck_signal >= drone_signal and worst_truck is not None:
            cust = worst_truck[1]
            res = _late_repair_best_reinsert_position(data, route, base_to_drone_customers, cust, ctx)
            if res is not None:
                if (res['best_late'] + eps) < best_late or (
                        abs(res['best_late'] - best_late) <= eps and res['best_cost'] + eps < best_cost):
                    route = res['best_route']
                    best_cost = res['best_cost']
                    best_late = res['best_late']
                    improved = True

        if (not improved) and worst_base is not None:
            b = int(worst_base[2])
            resb = _late_repair_best_base_reinsert(data, route, base_to_drone_customers, b, ctx)
            if resb is not None:
                if (resb['best_late'] + eps) < best_late or (
                        abs(resb['best_late'] - best_late) <= eps and resb['best_cost'] + eps < best_cost):
                    route = resb['best_route']
                    best_cost = resb['best_cost']
                    best_late = resb['best_late']
                    improved = True

        if not improved:
            break

    return route

def R_late_repair_reinsert(data, route, base_to_drone_customers, removed_customers, ctx):
    """
    迟到专项修复（建议低频使用）：
    - 先走主力 R_regret_then_drone
    - 仅当当前解仍有明显迟到（> LATE_REPAIR_TRIGGER）时，才触发 late_repair_truck_reinsert
    """
    res = R_regret_then_drone(data, route, base_to_drone_customers, removed_customers, ctx)
    if res is None:
        return None
    route2, b2d2 = res

    trigger = float(ctx.get("LATE_REPAIR_TRIGGER", 1e-6))
    # 快速判断是否有迟到（用同一评估口径）
    try:
        start_time = float(ctx.get("start_time", 0.0))
        truck_speed = float(ctx.get("truck_speed", sim.TRUCK_SPEED_UNITS))
        drone_speed = float(ctx.get("drone_speed", sim.DRONE_SPEED_UNITS))
        alpha_drone = float(ctx.get("alpha_drone", 0.3))
        lambda_late = float(ctx.get("lambda_late", 50.0))
        arrival_prefix = ctx.get("arrival_prefix", None)
        _, _, _, _, _, total_late, _ = sim.evaluate_truck_drone_with_time(
            data, route2, b2d2, start_time, truck_speed, drone_speed,
            alpha_drone, lambda_late, arrival_prefix=arrival_prefix
        )
        if float(total_late) <= trigger:
            return route2, b2d2
    except Exception:
        # 若评估失败，则不做 late_repair，保持稳健
        return route2, b2d2

    route3 = late_repair_truck_reinsert(data, route2, b2d2, ctx)
    return route3, b2d2

def nearest_neighbor_route_truck_only(data,
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

def _resolve_operator_list(names, g=None):
    """把字符串函数名解析成函数对象；优先在当前模块(operators)查找。"""
    # g 参数留着兼容，但实际上我们主要查 globals()
    current_globals = globals()
    ops_list = []
    for nm in names:
        fn = current_globals.get(nm, None)
        if fn is None or not callable(fn):
            raise RuntimeError(f"[CFG] 未找到算子函数: {nm} (请检查 operators.py)")
        ops_list.append(fn)
    return ops_list

def build_ab_cfg(cfg: dict):
    """把 cfg 中的字符串 DESTROYS/REPAIRS/ALLOWED_PAIRS 转成可执行函数对象。"""
    new_cfg = dict(cfg)

    # DESTROYS / REPAIRS
    if "DESTROYS" in new_cfg and new_cfg["DESTROYS"]:
        if isinstance(new_cfg["DESTROYS"][0], str):
            # 直接调用上面的 _resolve_operator_list，不需要传 g 了
            new_cfg["DESTROYS"] = _resolve_operator_list(new_cfg["DESTROYS"])

    if "REPAIRS" in new_cfg and new_cfg["REPAIRS"]:
        if isinstance(new_cfg["REPAIRS"][0], str):
            new_cfg["REPAIRS"] = _resolve_operator_list(new_cfg["REPAIRS"])

    # ALLOWED_PAIRS（paired 模式）
    if "ALLOWED_PAIRS" in new_cfg and new_cfg["ALLOWED_PAIRS"]:
        pairs = []
        for dnm, rnm in new_cfg["ALLOWED_PAIRS"]:
            # destroy
            if isinstance(dnm, str):
                D = globals().get(dnm)
                if D is None or not callable(D):
                    raise RuntimeError(f"[CFG] 未找到 destroy: {dnm}")
            elif callable(dnm):
                D = dnm
            else:
                raise RuntimeError(f"[CFG] destroy 既不是字符串也不是函数对象: {dnm}")

            # repair
            if isinstance(rnm, str):
                R = globals().get(rnm)
                if R is None or not callable(R):
                    raise RuntimeError(f"[CFG] 未找到 repair: {rnm}")
            elif callable(rnm):
                R = rnm
            else:
                raise RuntimeError(f"[CFG] repair 既不是字符串也不是函数对象: {rnm}")

            pairs.append((D, R))
        new_cfg["ALLOWED_PAIRS"] = pairs

    return new_cfg
# =========================================================
# 推荐算子组合（按“高接纳率 + 控成本/迟到”的目标）
# 说明：真正使用哪些算子由外部 cfg 决定，这里给出建议列表方便配置。
# =========================================================

RECOMMENDED_DESTROYS = [
    'D_reloc_focus_v2',
    'D_switch_coverage',
    'D_worst_route',
    # 低频探索：
    'D_random_route',
    # 低频控迟到：
    'D_late_worst',
]

RECOMMENDED_REPAIRS = [
    'R_regret_then_drone',
    'R_base_feasible_drone_first',
    # 低频控迟到：
    'R_late_repair_reinsert',
]

RECOMMENDED_ALLOWED_PAIRS = [
    ('D_reloc_focus_v2', 'R_regret_then_drone'),
    ('D_switch_coverage', 'R_base_feasible_drone_first'),
    ('D_worst_route', 'R_regret_then_drone'),
    ('D_random_route', 'R_regret_then_drone'),
    ('D_late_worst', 'R_late_repair_reinsert'),
]
