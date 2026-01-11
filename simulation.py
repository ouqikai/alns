# simulation.py
import math
import copy
from utils import compute_eta_map  # 如果 alns 依赖它，也可从这导

# =========================
# 全局仿真参数（带默认值）
# =========================
TRUCK_SPEED_KMH = 30.0
DRONE_SPEED_KMH = 60.0
SCALE_KM_PER_UNIT = 5.0
TRUCK_ROAD_FACTOR = 1.5

TRUCK_SPEED_UNITS = TRUCK_SPEED_KMH * SCALE_KM_PER_UNIT
DRONE_SPEED_UNITS = DRONE_SPEED_KMH * SCALE_KM_PER_UNIT
DRONE_RANGE_KM = 10.0
DRONE_RANGE_UNITS = DRONE_RANGE_KM * SCALE_KM_PER_UNIT
DRONE_CAPACITY = 2.0
NUM_DRONES_PER_BASE = 999  # 无限/充足无人机


def set_simulation_params(road_factor=None, truck_speed_kmh=None):
    """
    配置仿真参数的统一入口。
    主文件在 main 开头调用此函数来修改全局物理参数。
    """
    global TRUCK_ROAD_FACTOR, TRUCK_SPEED_KMH, TRUCK_SPEED_UNITS

    if road_factor is not None:
        TRUCK_ROAD_FACTOR = float(road_factor)
        if TRUCK_ROAD_FACTOR <= 0:
            raise ValueError(f"TRUCK_ROAD_FACTOR 必须 >0, got {TRUCK_ROAD_FACTOR}")

    if truck_speed_kmh is not None:
        TRUCK_SPEED_KMH = float(truck_speed_kmh)
        TRUCK_SPEED_UNITS = TRUCK_SPEED_KMH * SCALE_KM_PER_UNIT


def get_simulation_params():
    return {
        "TRUCK_ROAD_FACTOR": TRUCK_ROAD_FACTOR,
        "TRUCK_SPEED_UNITS": TRUCK_SPEED_UNITS,
        "DRONE_SPEED_UNITS": DRONE_SPEED_UNITS,
        "DRONE_RANGE_UNITS": DRONE_RANGE_UNITS,
        "NUM_DRONES_PER_BASE": NUM_DRONES_PER_BASE
    }


# =========================
# 核心物理计算函数
# =========================

def truck_arc_cost(data, i: int, j: int) -> float:
    """
    卡车弧距离（用于道路绕行/路况），统一口径。
    依赖模块级变量 TRUCK_ROAD_FACTOR。
    """
    return float(data.costMatrix[i, j]) * float(TRUCK_ROAD_FACTOR)


def compute_truck_schedule(data, route, start_time: float = 0.0, speed: float = None):
    """
    计算卡车在各节点的到达时间和总行驶时间（小时）。
    """
    if speed is None:
        speed = TRUCK_SPEED_UNITS

    arrival_times = {}
    current_time = start_time

    if not route:
        return arrival_times, current_time, 0.0

    # 起点：中心仓库或虚拟点
    start_idx = route[0]
    arrival_times[start_idx] = current_time
    prev = start_idx

    for pos in range(1, len(route)):
        curr = route[pos]
        # 使用本模块的 truck_arc_cost
        dist = truck_arc_cost(data, prev, curr)
        travel_time = dist / speed
        current_time += travel_time

        if pos < len(route) - 1:
            arrival_times[curr] = current_time
        prev = curr

    total_time = current_time

    # 计算迟到
    total_late = 0.0
    for idx, node in enumerate(data.nodes):
        if idx not in arrival_times:
            continue
        nt = str(node.get('node_type', '')).strip().lower()
        if not ((nt == 'customer') or (nt == 'c') or ('cust' in nt)):
            continue
        arr = float(arrival_times[idx])
        due = float(node.get('due_time', float('inf')))
        if arr > due:
            total_late += (arr - due)

    return arrival_times, total_time, total_late


def compute_multi_drone_schedule(data,
                                 base_to_drone_customers,
                                 arrival_times,
                                 num_drones_per_base: int = None,
                                 drone_speed: float = None,
                                 arrival_prefix: dict = None,
                                 default_base_arrival: float = 0.0):
    """多无人机调度（List Scheduling）。"""
    if num_drones_per_base is None:
        num_drones_per_base = NUM_DRONES_PER_BASE
    if drone_speed is None:
        drone_speed = DRONE_SPEED_UNITS

    depart_times = {}
    finish_times = {}
    base_finish_times = {}

    for base_idx, clients in base_to_drone_customers.items():
        if not clients:
            t_base = arrival_times.get(base_idx, None)
            if t_base is None and arrival_prefix is not None:
                t_base = arrival_prefix.get(base_idx, None)
            if t_base is None:
                t_base = float(default_base_arrival)
            base_finish_times[base_idx] = float(t_base)
            continue

        base_arrival = arrival_times.get(base_idx, None)
        if base_arrival is None and arrival_prefix is not None:
            base_arrival = arrival_prefix.get(base_idx, None)
        if base_arrival is None:
            base_arrival = float(default_base_arrival)

        drone_available = [float(base_arrival)] * int(num_drones_per_base)

        for c in clients:
            k = min(range(int(num_drones_per_base)), key=lambda i: drone_available[i])
            depart = float(drone_available[k])

            d_bc = float(data.costMatrix[base_idx, c])
            fly_time = 2.0 * d_bc / float(drone_speed)
            finish = depart + fly_time

            depart_times[c] = depart
            finish_times[c] = finish
            drone_available[k] = finish

        base_finish_times[base_idx] = max(drone_available)

    return depart_times, finish_times, base_finish_times


def _compose_cost(truck_dist, drone_dist, total_late, alpha_drone, lambda_late):
    truck_dist = float(truck_dist)
    drone_dist = float(drone_dist)
    total_late = float(total_late)
    total_cost = truck_dist + float(alpha_drone) * drone_dist + float(lambda_late) * total_late
    return total_cost, truck_dist


def evaluate_truck_drone_with_time(data, route, base_to_drone, start_time=0.0, truck_speed=None,
                                   drone_speed=None, alpha_drone=0.3, lambda_late=50.0,
                                   arrival_prefix: dict = None):
    """粗粒度评估（用于 ALNS 内部快速比较）。"""
    if truck_speed is None:
        truck_speed = TRUCK_SPEED_UNITS
    if drone_speed is None:
        drone_speed = DRONE_SPEED_UNITS

    arrival_times, total_time, _ = compute_truck_schedule(data, route, start_time=start_time, speed=truck_speed)

    truck_dist = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        truck_dist += truck_arc_cost(data, a, b)

    drone_dist = 0.0
    drone_late = 0.0
    for base_idx, clients in base_to_drone.items():
        base_arrival = arrival_times.get(base_idx, None)
        if base_arrival is None and arrival_prefix is not None:
            base_arrival = arrival_prefix.get(base_idx, None)
        if base_arrival is None:
            base_arrival = float(start_time)

        for c in clients:
            d = float(data.costMatrix[base_idx, c])
            drone_dist += 2.0 * d
            service_time = base_arrival + (2.0 * d / float(drone_speed))
            due = float(data.nodes[c].get('due_time', float('inf')))
            if service_time > due:
                drone_late += (service_time - due)

    truck_late = 0.0
    for idx in route:
        if data.nodes[idx].get('node_type') == 'customer':
            due = float(data.nodes[idx].get('due_time', float('inf')))
            arr = float(arrival_times.get(idx, float('inf')))
            if arr > due:
                truck_late += (arr - due)

    total_late = truck_late + drone_late
    total_cost, _ = _compose_cost(truck_dist, drone_dist, total_late, alpha_drone, lambda_late)

    return total_cost, truck_dist, drone_dist, truck_late, drone_late, total_late, total_time


def evaluate_full_system(data, full_route, full_b2d,
                         alpha_drone=0.3, lambda_late=50.0,
                         truck_speed=None, drone_speed=None):
    """全系统口径评估。"""
    if truck_speed is None:
        truck_speed = TRUCK_SPEED_UNITS
    if drone_speed is None:
        drone_speed = DRONE_SPEED_UNITS

    truck_dist = 0.0
    for i in range(len(full_route) - 1):
        truck_dist += truck_arc_cost(data, full_route[i], full_route[i + 1])

    arrival, total_time, truck_late = compute_truck_schedule(
        data, full_route, start_time=0.0, speed=truck_speed
    )

    depart, finish, base_finish = compute_multi_drone_schedule(
        data, full_b2d, arrival,
        num_drones_per_base=NUM_DRONES_PER_BASE,
        drone_speed=drone_speed
    )

    drone_dist = 0.0
    drone_late = 0.0
    for b, cs in full_b2d.items():
        for c in cs:
            d_bc = data.costMatrix[b, c]
            drone_dist += 2.0 * d_bc
            arrive_c = depart[c] + d_bc / drone_speed
            due = data.nodes[c].get('due_time', float('inf'))
            if arrive_c > due:
                drone_late += (arrive_c - due)

    total_late = truck_late + drone_late
    system_time = max(total_time, max(base_finish.values()) if base_finish else 0.0)

    cost, truck_dist_eff = _compose_cost(truck_dist, drone_dist, total_late, alpha_drone, lambda_late)

    return {
        "cost": cost,
        "system_time": system_time,
        "truck_dist": truck_dist,
        "truck_dist_eff": truck_dist_eff,
        "drone_dist": drone_dist,
        "truck_late": truck_late,
        "drone_late": drone_late,
        "total_late": total_late,
        "arrival": arrival,
        "depart": depart,
        "finish": finish,
        "base_finish": base_finish,
        "truck_total_time": total_time,
    }


def classify_clients_for_drone(data, allowed_customers=None, feasible_bases=None):
    """
    客户分类逻辑：分为必须卡车 vs 可无人机。
    """
    # 依赖 DRONE_RANGE_UNITS

    base_indices = [i for i, n in enumerate(data.nodes) if n['node_type'] == 'base']
    if feasible_bases is not None:
        feasible_set = set(int(b) for b in feasible_bases)
        base_indices = [b for b in base_indices if b in feasible_set]

    central_idx = data.central_idx
    if central_idx not in base_indices:
        base_indices.append(central_idx)

    customer_indices = [i for i, n in enumerate(data.nodes) if n['node_type'] == 'customer']
    if allowed_customers is not None:
        allowed_set = set(allowed_customers)
        customer_indices = [c for c in customer_indices if c in allowed_set]

    base_to_drone_customers = {b: [] for b in base_indices}
    truck_customers = []

    for c in customer_indices:
        node_c = data.nodes[c]
        if node_c.get('force_truck', 0) == 1:
            truck_customers.append(c)
            continue

        b_lock = node_c.get('base_lock', None)
        if b_lock is not None:
            try:
                b_lock = int(b_lock)
            except Exception:
                b_lock = None

        if b_lock is not None and b_lock in base_indices:
            best_base = b_lock
            best_dist = float(data.costMatrix[best_base, c])
        else:
            best_base = None
            best_dist = float('inf')
            for b in base_indices:
                d = float(data.costMatrix[b, c])
                if d < best_dist:
                    best_dist = d
                    best_base = b

        if best_base is not None and 2.0 * best_dist <= DRONE_RANGE_UNITS:
            base_to_drone_customers[best_base].append(c)
        else:
            truck_customers.append(c)

    return base_to_drone_customers, truck_customers

def check_disjoint(data, route, base_to_drone_customers):
    """护栏：检查互斥性"""
    route_customers = {i for i in route if data.nodes[i]['node_type'] == 'customer'}
    drone_customers = set()
    for _, lst in base_to_drone_customers.items():
        for c in lst:
            drone_customers.add(c)
    inter = route_customers & drone_customers
    if inter:
        print("警告：以下客户同时出现在卡车路径和无人机列表中：", [data.nodes[i]['node_id'] for i in inter])


def sanity_check_full(data, full_route, full_b2d):
    """一致性检查"""
    all_customers = {i for i, n in enumerate(data.nodes) if n.get('node_type') == 'customer'}
    truck_set = {i for i in full_route if i in all_customers}
    drone_list = [c for cs in full_b2d.values() for c in cs]
    drone_set = set(drone_list)
    if len(drone_list) != len(drone_set):
        raise RuntimeError(f"[FULL-check] drone 列表有重复")
    inter = truck_set & drone_set
    if inter:
        raise RuntimeError(f"[FULL-check] truck/drone 客户冲突")
    covered = truck_set | drone_set
    if len(covered) > len(all_customers):
        raise RuntimeError(f"[FULL-check] 覆盖数超过客户总数")
    return {
        "n_customers": len(all_customers),
        "truck_customers": len(truck_set),
        "drone_customers": len(drone_set),
        "covered": len(covered),
    }

def guardrail_check(data, full_route, full_b2d, tag=""):
    """工程护栏"""
    all_customers = {i for i, n in enumerate(data.nodes) if n.get("node_type") == "customer"}
    truck_customers = {i for i in full_route if
                       0 <= i < len(data.nodes) and data.nodes[i].get("node_type") == "customer"}
    drone_customers = {c for cs in full_b2d.values() for c in cs}

    covered = truck_customers | drone_customers
    uncovered = all_customers - covered
    if uncovered:
        nids = [data.nodes[i]["node_id"] for i in sorted(uncovered)]
        raise RuntimeError(f"[GUARDRAIL]{tag} 存在未覆盖客户 uncovered={len(uncovered)} node_id={nids}")

    inter = truck_customers & drone_customers
    if inter:
        raise RuntimeError(f"[GUARDRAIL]{tag} 冲突")