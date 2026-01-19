# operators.py
import random
import math
import simulation as sim


# =========================================================
# 调试辅助（由 dynamic_logic.alns_truck_drone 透传 ctx）
# =========================================================
def _dbg_should_print(ctx: dict) -> bool:
    """中文注释：仅在 dbg_alns=True 且命中节流条件时打印。"""
    try:
        if not bool(ctx.get("dbg_alns", False)):
            return False
        it = int(ctx.get("_dbg_it", 0))
        every = max(1, int(ctx.get("_dbg_every", 50)))
        return (it == 1) or (it % every == 0)
    except Exception:
        return False


def _dbg_repair_summary(data, route, b2d, ctx, tag: str):
    if not _dbg_should_print(ctx):
        return
    try:
        cust_set = set(getattr(data, "customer_indices", []))
        truck_set = {int(c) for c in route if int(c) in cust_set}
        drone_set = set()
        for _b, _cs in (b2d or {}).items():
            for _c in _cs:
                drone_set.add(int(_c))
        removed_n = int(ctx.get("num_remove", 0))
        dname = str(ctx.get("_dbg_d_name", ""))
        rname = str(ctx.get("_dbg_r_name", ""))
        print(f"[DBG-REPAIR] it={ctx.get('_dbg_it')} D={dname} R={rname} {tag}: "
              f"truck={len(truck_set)} drone={len(drone_set)} removed_req={removed_n}")
    except Exception:
        pass


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
    num_remove = ctx["num_remove"]
    protected = ctx.get("protected_nodes", set())
    destroyed_route, removed = random_removal(route, num_remove, data, protected=protected)
    destroyed_b2d = {b: lst[:] for b, lst in b2d.items()}
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

    pool_main = (moved_acc | force_set | boundary)
    pool_rej = moved_rej - pool_main

    def is_removable(i):
        return (0 <= i < len(data.nodes)
                and data.nodes[i].get("node_type") == "customer"
                and i not in protected)

    pool_main = [i for i in pool_main if is_removable(i)]
    pool_rej = [i for i in pool_rej if is_removable(i)]

    removed = []

    if pool_main:
        take = min(num_remove, len(pool_main))
        removed.extend(random.sample(pool_main, take))

    if len(removed) < num_remove and pool_rej:
        need = num_remove - len(removed)
        cap = max(1, num_remove // 3)
        take = min(need, cap, len(pool_rej))
        removed.extend(random.sample(pool_rej, take))

    route_pos = {node: idx for idx, node in enumerate(route)}
    extra = []
    for c in list(removed):
        if c not in route_pos:
            continue
        j = route_pos[c]
        for nb in [route[j - 1] if j - 1 >= 0 else None, route[j + 1] if j + 1 < len(route) else None]:
            if nb is None:
                continue
            if is_removable(nb) and nb not in removed and nb not in extra:
                extra.append(nb)

    random.shuffle(extra)
    for nb in extra:
        if len(removed) >= num_remove:
            break
        removed.append(nb)

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
        # 尝试兜底
        drone_range = sim.DRONE_RANGE_UNITS

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
        feas = sim.feasible_bases_for_customer(data, i, ctx, route_set, drone_range)
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


# =========================================================
# 无人机修复辅助 & 具体 Repair 算子
# =========================================================

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
    _dbg_repair_summary(data, r, bd, ctx, tag="greedy_only")
    return r, bd


def R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r = regret_insert(data, destroyed_route, removed_customers)
    bd = {b: lst[:] for b, lst in destroyed_b2d.items()}
    _dbg_repair_summary(data, r, bd, ctx, tag="regret_only")
    return r, bd


def R_greedy_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r, bd = R_greedy_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    _dbg_repair_summary(data, r, bd, ctx, tag="greedy_then_drone")
    return r, bd


def R_regret_then_drone(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    r, bd = R_regret_only(data, destroyed_route, destroyed_b2d, removed_customers, ctx)
    r, bd = drone_repair_feasible(data, r, bd, ctx, k_moves=8, sample_k=10)
    _dbg_repair_summary(data, r, bd, ctx, tag="regret_then_drone")
    return r, bd


def R_base_feasible_drone_first(data, destroyed_route, destroyed_b2d, removed_customers, ctx):
    bases_to_visit = ctx.get("bases_to_visit", [])
    drone_range = ctx.get("drone_range", None)
    if drone_range is None:
        drone_range = sim.DRONE_RANGE_UNITS

    route = destroyed_route[:]
    b2d = {b: lst[:] for b, lst in destroyed_b2d.items()}
    route_set = set(route)
    force_set = set(ctx.get("force_truck_set", set()))

    for cid in removed_customers:
        if cid in force_set:
            continue
        feas = sim.feasible_bases_for_customer(data, cid, ctx, route_set, drone_range)
        lockb = data.nodes[cid].get("base_lock", None)
        if lockb is not None:
            feas = [b for b in feas if b == lockb]
        if feas:
            b = random.choice(feas)
            b2d.setdefault(b, [])
            b2d[b].append(cid)
        else:
            route = greedy_insert(data, route, [cid])
            route_set = set(route)

    _dbg_repair_summary(data, route, b2d, ctx, tag="base_feasible_drone_first")
    return route, b2d


# =========================================================
# Late Repair (复杂修复)
# =========================================================

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
    res = R_regret_then_drone(data, route, base_to_drone_customers, removed_customers, ctx)
    if res is None:
        return None
    route2, b2d2 = res
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