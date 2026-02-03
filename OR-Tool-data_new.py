# -*- coding: utf-8 -*-
"""
数据集格式（nodes.csv + 可选 events.csv）：

nodes.csv 字段：
NODE_ID,NODE_TYPE,ORIG_X,ORIG_Y,DEMAND,READY_TIME,DUE_TIME

events.csv 字段（离线“请求脚本”，仅事实，不含 ACCEPT/REJECT）：
EVENT_ID,EVENT_TIME,NODE_ID,NEW_X,NEW_Y,EVENT_CLASS,DELTA_AVAIL_H

重要说明：
- NODE_ID：客户/设施的编号（求解端同口径）。
- EVENT_ID：事件序号（仅用于唯一标识事件、便于排序/调试），不是客户编号。
- EVENT_CLASS 仅用于标注请求类型（IN_DB / CROSS_DB / OUT_DB），不代表接受或拒绝。
- DELTA_AVAIL_H：相对可用时长（小时），求解端用它计算 L=max(PROM_DUE, PROM_READY+DELTA_AVAIL_H)。
- 坐标单位：5单位 = 1km（可在 GenConfig.units_per_km 修改）
- 时间单位：小时
- 本脚本不再生成时间窗（READY_TIME/DUE_TIME 置 0），时间窗由求解端在场景0生成并冻结为“平台承诺窗”。
- 路况系数口径与求解端一致：卡车“距离=欧氏距离×truck_road_factor”，速度保持 truck_speed_kmh 不变。

运行后输出：nodes_<N>_seed<seed>_<timestamp>.csv 以及 events_<N>_seed<seed>_<timestamp>.csv
"""

import csv
import math
import random
import datetime
import json
import hashlib
import sys
import platform
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

# =========================
# 基础工具
# =========================
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sample_in_annulus(cx: float, cy: float, r_min: float, r_max: float) -> Tuple[float, float]:
    """在环带 [r_min, r_max] 内均匀采样（面积均匀）"""
    theta = random.random() * 2.0 * math.pi
    r = r_min + math.sqrt(random.random()) * (r_max - r_min)
    return cx + r * math.cos(theta), cy + r * math.sin(theta)

def fmt(v):
    """CSV：数值保留2位"""
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        return f"{float(v):.2f}"
    return str(v)

# =========================
# 参数
# =========================
@dataclass
class GenConfig:
    n_customers: int = 100

    # 坐标与比例
    visual_range: float = 100.0
    units_per_km: float = 5.0

    # 速度 km/h
    truck_speed_kmh: float = 30.0
    # 卡车路况系数：用于“卡车距离=欧氏距离×系数”（求解端口径）；
    # 生成时间窗时用等效慢速：v_eff = v / truck_road_factor
    truck_road_factor: float = 1.5

    drone_speed_kmh: float = 60.0
    drones_per_base: int = 3

    # 无人机往返续航 km（覆盖判断用单程半径=往返/2）
    drone_roundtrip_km: float = 10.0

    # 基站数量：按 12%（25->3,50->6,...）
    base_ratio: float = 0.12
    min_bases: int = 1
    # 中文注释：不同规模下的基站数量覆盖表；未命中则按 base_ratio 计算
    base_count_override: dict = None  # 先给默认

    # 圈外卡车客户占比
    truck_customer_ratio: float = 0.20

    # 美观：最小距离（单位）
    min_dist_global: float = 7.0
    min_dist_within_ring: float = 10.0

    # 环带范围（相对单程半径R）
    ring_min_ratio: float = 0.25
    ring_max_ratio: float = 0.90

    # 圈内需求：80% 1-3；20% 4-5
    drone_small_prob: float = 0.80
    drone_small_range: Tuple[int, int] = (1, 3)
    drone_large_range: Tuple[int, int] = (4, 5)

    # 圈外需求：70% 6-8；30% 9-10
    truck_mid_prob: float = 0.70
    truck_mid_range: Tuple[int, int] = (6, 8)
    truck_large_range: Tuple[int, int] = (9, 10)

    # 时间窗（小时）
    center_slack_max: float = 0.25  # 时间窗中心扰动
    window_width: float = 0.30      # 18min（基础宽度）

    # 时间窗自动放宽（客户数越大，时间窗越松；并对“设施到达时间估计”做整体后移）
    tw_width_scale_per_100: float = 0.5  # n=100 => width*(1+0.5)=1.5倍；n=r_200 => 2.0倍
    tw_global_shift_ratio: float = 0.15  # 预计系统工期 T_hat 的比例，用于整体后移 time-window 中心
    # =========================
    # 最终版时间窗参数（小时）
    # 说明：本版本采用“参考到达时刻锚定”的简单规则：
    #   - 卡车客户：ready=t_truck_arrival_ref；due=ready+truck_due_plus_h
    #   - 无人机客户：ready=t_base_arrival_ref + wait_est + fly_one；due=ready+drone_due_plus_h
    # 其中 wait_est 为粗估等待（同基站共享）：wait_est = (sum_c fly_one(b,c)) / drones_per_base
    truck_due_plus_h: float = 1.0
    drone_due_plus_h: float = 2.0

    seed: int = 2

    def __post_init__(self):
        """根据客户规模自动扩展坐标范围。
        规则：每增加 100 个客户，坐标边长 +100。
        - n<100  => 100×100
        - 100-199=> r_200×r_200
        - r_200-299=> 300×300
        说明：若用户手动设置 visual_range 更大，则保留更大的值。
        """
        # auto_range = 100.0 * (int(self.n_customers) // 100 + 1)
        # self.visual_range = max(float(self.visual_range), auto_range)
        pass
# =========================
# 固定设施：central + bases
# =========================
def generate_fixed_bases_and_central(cfg: GenConfig) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """central在中心；bases做均匀网格取前K个"""
    central = (cfg.visual_range / 2.0, cfg.visual_range / 2.0)
    k = max(cfg.min_bases, int(round(cfg.n_customers * cfg.base_ratio)))
    # 若当前规模在覆盖表里，就用指定基站数
    if hasattr(cfg, "base_count_override") and cfg.base_count_override and (cfg.n_customers in cfg.base_count_override):
        k = int(cfg.base_count_override[cfg.n_customers])

    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))

    margin = cfg.visual_range * 0.15
    xs = [margin + i * (cfg.visual_range - 2 * margin) / (cols - 1 if cols > 1 else 1) for i in range(cols)]
    ys = [margin + j * (cfg.visual_range - 2 * margin) / (rows - 1 if rows > 1 else 1) for j in range(rows)]

    bases = []
    for y in ys:
        for x in xs:
            if len(bases) >= k:
                break
            p = (float(x), float(y))
            if distance(p, central) < 1e-6:
                continue
            bases.append(p)
        if len(bases) >= k:
            break
    return bases, central

# =========================
# 需求采样
# =========================
def sample_drone_demand(cfg: GenConfig) -> int:
    if random.random() < cfg.drone_small_prob:
        return random.randint(cfg.drone_small_range[0], cfg.drone_small_range[1])
    return random.randint(cfg.drone_large_range[0], cfg.drone_large_range[1])

def sample_truck_demand(cfg: GenConfig) -> int:
    if random.random() < cfg.truck_mid_prob:
        return random.randint(cfg.truck_mid_range[0], cfg.truck_mid_range[1])
    return random.randint(cfg.truck_large_range[0], cfg.truck_large_range[1])

# =========================
# 客户生成
# =========================
def generate_base_clients(
        bases: List[Tuple[float, float]],
        central: Tuple[float, float],
        drone_oneway_units: float,
        n_base_customers: int,
        cfg: GenConfig
) -> List[Tuple[float, float, int]]:
    """圈内客户（设施附近环带），最小距离约束"""
    facilities = bases + [central]
    r_min = drone_oneway_units * cfg.ring_min_ratio
    r_max = drone_oneway_units * cfg.ring_max_ratio

    clients: List[Tuple[float, float, int]] = []
    global_xy: List[Tuple[float, float]] = []

    per = n_base_customers // len(facilities)
    extra = n_base_customers % len(facilities)

    for j, (fx, fy) in enumerate(facilities):
        need = per + (1 if j < extra else 0)
        placed_local: List[Tuple[float, float]] = []

        for _ in range(need):
            last_xy = (fx, fy)
            for _attempt in range(300):
                x, y = sample_in_annulus(fx, fy, r_min, r_max)
                x = clamp(x, 0.0, cfg.visual_range)
                y = clamp(y, 0.0, cfg.visual_range)
                last_xy = (x, y)

                if any(distance((x, y), p) < cfg.min_dist_within_ring for p in placed_local):
                    continue
                if any(distance((x, y), p) < cfg.min_dist_global for p in global_xy):
                    continue
                break

            x, y = last_xy
            dem = sample_drone_demand(cfg)
            clients.append((x, y, dem))
            placed_local.append((x, y))
            global_xy.append((x, y))

    return clients

def generate_truck_clients_outside_coverage(
        bases: List[Tuple[float, float]],
        central: Tuple[float, float],
        drone_oneway_units: float,
        n_truck_customers: int,
        cfg: GenConfig,
        existing_xy: List[Tuple[float, float]]
) -> List[Tuple[float, float, int]]:
    """圈外卡车客户：必须在所有覆盖圈外"""
    facilities = bases + [central]
    truck_clients: List[Tuple[float, float, int]] = []
    tries = 0
    max_tries = 200000

    while len(truck_clients) < n_truck_customers and tries < max_tries:
        tries += 1
        x = random.uniform(0.0, cfg.visual_range)
        y = random.uniform(0.0, cfg.visual_range)

        if any(distance((x, y), f) <= drone_oneway_units for f in facilities):
            continue
        if any(distance((x, y), p) < cfg.min_dist_global for p in existing_xy):
            continue
        if any(distance((x, y), (tx, ty)) < cfg.min_dist_global for tx, ty, _ in truck_clients):
            continue

        dem = sample_truck_demand(cfg)
        truck_clients.append((x, y, dem))
        existing_xy.append((x, y))

    if len(truck_clients) < n_truck_customers:
        print(f"[WARNING] 圈外卡车客户不足 {len(truck_clients)}/{n_truck_customers}，可调小 min_dist 或减小覆盖半径")
    return truck_clients
def plan_truck_route_nearest_neighbor(
        bases: List[Tuple[float, float]],
        central: Tuple[float, float],
        truck_clients: List[Tuple[float, float, int]]
) -> List[Tuple[float, float]]:
    """central + bases + truck_clients 的 NN 路径，仅用于生成到达时间特征"""
    nodes = [central] + bases + [(x, y) for x, y, _ in truck_clients]
    visited = [False] * len(nodes)
    route_idx = [0]
    visited[0] = True
    cur = 0

    while len(route_idx) < len(nodes):
        nxt, best = None, float("inf")
        for j in range(len(nodes)):
            if not visited[j]:
                d = distance(nodes[cur], nodes[j])
                if d < best:
                    best, nxt = d, j
        visited[nxt] = True
        route_idx.append(nxt)
        cur = nxt

    route_idx.append(0)
    return [nodes[i] for i in route_idx]

# =========================
# 用于时间窗：粗略生成“卡车到达设施时间”
def plan_truck_route_ortools_for_tw(
        bases: List[Tuple[float, float]],
        central: Tuple[float, float],
        truck_clients: List[Tuple[float, float, int]],
        time_limit_s: int = 2,
        seed: int = 1,
        use_local_search: bool = True,
        fallback_to_nn: bool = False,
        dist_scale: int = 1000
) -> List[Tuple[float, float]]:
    """
    中文注释：
    用 OR-Tools 求一条 TSP 路径（central + bases + truck_clients），仅用于“生成时间窗的参考到达时刻”。
    - 返回 route_coords：[(x,y), ...]，起点=central，终点=central，且每个点访问一次
    - use_local_search=True 会用局部搜索提升路线质量（更贴近“合理路径”）
    - dist_scale 把浮点距离转成整数（OR-Tools 要求 arc cost 是 int）
    """
    # 组装节点：0号为 central，后面依次 bases，再后面圈外卡车客户坐标
    nodes = [central] + list(bases) + [(x, y) for x, y, _ in truck_clients]
    n = len(nodes)
    if n <= 2:
        return [central, central]

    try:
        # 中文注释：OR-Tools 路由求解器
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        import ortools
    except Exception as e:
        if fallback_to_nn:
            print(f"[TW-ROUTE] OR-Tools 未安装或导入失败，回退到最近邻NN。err={e}")
            return plan_truck_route_nearest_neighbor(bases, central, truck_clients)
        raise RuntimeError(f"[TW-ROUTE] OR-Tools 未安装或导入失败：{e}") from e

    # 构造整数距离矩阵
    def int_cost(i: int, j: int) -> int:
        # 中文注释：四舍五入到整数，避免 0 距离导致退化
        d = distance(nodes[i], nodes[j])
        return int(max(1, round(d * dist_scale)))

    dist_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_mat[i][j] = 0
            else:
                dist_mat[i][j] = int_cost(i, j)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1辆车，起终点都为 0（central）
    routing = pywrapcp.RoutingModel(manager)

    # 代价回调
    def dist_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return dist_mat[i][j]

    transit_cb_idx = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # 求解参数
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    # 中文注释：为了更贴近“较优路径”，可开局部搜索；关闭则更确定但质量略差
    if use_local_search:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    # 时间限制 + 随机种子（便于复现）
    params.time_limit.FromSeconds(int(time_limit_s))
    try:
        params.random_seed = int(seed)
    except Exception:
        pass
    params.log_search = False

    sol = routing.SolveWithParameters(params)
    if sol is None:
        if fallback_to_nn:
            print("[TW-ROUTE] OR-Tools 未找到可行解，回退到最近邻NN。")
            return plan_truck_route_nearest_neighbor(bases, central, truck_clients)
        raise RuntimeError("[TW-ROUTE] OR-Tools 未找到可行解（SolveWithParameters returned None）。")

    # 读取解：节点访问顺序（包含回到 depot）
    order = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(manager.IndexToNode(idx))  # end（也是 0）

    route_coords = [nodes[k] for k in order]

    # 中文注释：把 OR-Tools 版本打印出来，方便你记录进 meta.json（可选）
    try:
        print(f"[TW-ROUTE] OR-Tools route ok. n={n}, time_limit={time_limit_s}s, seed={seed}, ortools={ortools.__version__}")
    except Exception:
        print(f"[TW-ROUTE] OR-Tools route ok. n={n}, time_limit={time_limit_s}s, seed={seed}")

    return route_coords

def calculate_arrival_times(route_coords: List[Tuple[float, float]], truck_speed_units_per_h: float) -> Dict[Tuple[float, float], float]:
    arrival = {route_coords[0]: 0.0}
    t = 0.0
    for i in range(len(route_coords) - 1):
        a, b = route_coords[i], route_coords[i + 1]
        t += distance(a, b) / truck_speed_units_per_h
        # 若同一坐标被多次经过（例如 central 回程），只保留最早到达时间
        arrival[b] = min(arrival.get(b, float('inf')), t)
    return arrival

def nearest_facility(pos: Tuple[float, float], bases: List[Tuple[float, float]], central: Tuple[float, float]) -> Tuple[float, float]:
    facilities = bases + [central]
    return min(facilities, key=lambda f: distance(pos, f))

def estimate_makespan_hours(cfg: GenConfig) -> float:
    """粗略估计系统工期（小时），用于生成更合理的时间窗中心（整体后移）。"""
    side = float(cfg.visual_range)
    n = max(1.0, float(cfg.n_customers))
    A = side * side
    tsp_len = 0.712 * math.sqrt(n * A)  # 期望路长（单位与坐标一致）
    truck_speed_units_per_h = float(cfg.truck_speed_kmh) * float(cfg.units_per_km) / max(1e-9, float(cfg.truck_road_factor))
    if truck_speed_units_per_h <= 1e-9:
        return 0.0
    return tsp_len / truck_speed_units_per_h

def make_time_window(center_t: float, cfg: GenConfig) -> Tuple[float, float]:
    """(ready, due)：以 center_t 为中心，宽度随规模放宽"""
    t_center = float(center_t) + random.uniform(0.0, cfg.center_slack_max)
    w = cfg.window_width * (1.0 + float(getattr(cfg, 'tw_width_scale_per_100', 0.0)) * (float(cfg.n_customers) / 100.0))
    ready = max(0.0, t_center - w / 2.0)
    due = t_center + w / 2.0
    return ready, due


def make_time_window_from_ready(ready0: float, cfg: GenConfig) -> Tuple[float, float]:
    """锚定“参考最早服务时刻”的时间窗生成规则（可复现）。

    规则（你老师那套口径）：
      ready = ready0 + wait，其中 wait ~ U(0, center_slack_max)
      due   = ready + width（width 随规模放宽）
    其中 ready0 对应：
      - 无人机客户：ready0 = t_base_arrival + fly_one_way
      - 卡车客户：ready0 = t_truck_arrival
    """
    wait = random.uniform(0.0, cfg.center_slack_max)
    ready = max(0.0, float(ready0) + wait)

    w = cfg.window_width * (1.0 + float(getattr(cfg, 'tw_width_scale_per_100', 0.0)) * (float(cfg.n_customers) / 100.0))
    due = ready + w
    return ready, due


# =========================
# CSV 写出
# =========================
def write_csv(
        csv_file: str,
        bases: List[Tuple[float, float]],
        central: Tuple[float, float],
        base_customers: List[Tuple[float, float, int]],
        truck_customers: List[Tuple[float, float, int]],
        route_coords: List[Tuple[float, float]],
        cfg: GenConfig,
        truck_speed_units_per_h: float
):
    fieldnames = [
        "NODE_ID", "NODE_TYPE",
        "ORIG_X", "ORIG_Y",
        "DEMAND",
        "READY_TIME", "DUE_TIME",
    ]

    # 注意：卡车行驶时间需体现 road_factor（与求解端口径一致）
    truck_speed_units_per_h_eff = float(truck_speed_units_per_h) / max(1e-9, float(cfg.truck_road_factor))
    arrival_times = calculate_arrival_times(route_coords, truck_speed_units_per_h_eff)
    # 中文注释：调试用——打印生成端用于时间窗的“各基站预估到达时刻”
    print("[GEN-ARR] truck_speed_eff=", truck_speed_units_per_h_eff)
    for bi, bpos in enumerate(bases, start=1):
        t = arrival_times.get(bpos, None)
        print(f"[GEN-ARR] base_id={bi} pos={bpos} t_arr={t}")

    drone_speed_units_per_h = cfg.drone_speed_kmh * cfg.units_per_km
    eps = 1e-9

    # =========================
    # 最终版时间窗：无人机客户粗估等待（生成端不做精细调度）
    # wait_est_b = (sum_{c∈C_b} fly_one(b,c)) / drones_per_base
    # 其中 fly_one(b,c) 为单程飞行时间（小时），C_b 为“最近设施= b”的无人机客户集合
    # =========================
    facilities = bases + [central]
    m_dr = max(1, int(getattr(cfg, "drones_per_base", 1)))
    sum_fly_one_by_fac = {pos: 0.0 for pos in facilities}
    cnt_by_fac = {pos: 0 for pos in facilities}
    for (x_d, y_d, _dem_d) in base_customers:
        fac_pos = nearest_facility((x_d, y_d), bases, central)
        fly_one_h = distance((x_d, y_d), fac_pos) / max(drone_speed_units_per_h, eps)
        sum_fly_one_by_fac[fac_pos] = sum_fly_one_by_fac.get(fac_pos, 0.0) + float(fly_one_h)
        cnt_by_fac[fac_pos] = cnt_by_fac.get(fac_pos, 0) + 1
    wait_est_by_fac = {pos: (float(sum_fly_one_by_fac.get(pos, 0.0)) / float(m_dr)) for pos in facilities}

    # 中文注释：调试打印——每个基站的无人机客户数与粗估等待（小时）
    print(f"[GEN-WAIT] drones_per_base={m_dr}")
    for bi, bpos in enumerate(bases, start=1):
        print(f"[GEN-WAIT] base_id={bi} n_drone={cnt_by_fac.get(bpos, 0)} wait_est={wait_est_by_fac.get(bpos, 0.0):.3f}h")
    print(f"[GEN-WAIT] central n_drone={cnt_by_fac.get(central, 0)} wait_est={wait_est_by_fac.get(central, 0.0):.3f}h")


    node_id = 0
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # central
        w.writerow({
            "NODE_ID": node_id,
            "NODE_TYPE": "central",
            "ORIG_X": fmt(central[0]), "ORIG_Y": fmt(central[1]),
            "DEMAND": fmt(0.0),
            "READY_TIME": fmt(0.0),
            "DUE_TIME": fmt(0.0),
        })
        node_id += 1

        # bases
        for b in bases:
            t_b = arrival_times.get(b, 0.0)
            w.writerow({
                "NODE_ID": node_id,
                "NODE_TYPE": "base",
                "ORIG_X": fmt(b[0]), "ORIG_Y": fmt(b[1]),
                "DEMAND": fmt(0.0),
                "READY_TIME": fmt(t_b),
                "DUE_TIME": fmt(t_b),
            })
            node_id += 1

        # 中文注释：最终版时间窗规则不做整体后移（避免引入额外超参数）
        shift_h = 0.0

        def write_customer(x, y, dem, is_truck: bool):
            nonlocal node_id
            fac_pos = nearest_facility((x, y), bases, central)
            t_fac = float(arrival_times.get(fac_pos, 0.0))  # 参考路线下到达该设施的时刻（小时）
            # =========================
            # [PROMISE] 时间窗不在生成端设定：READY_TIME/DUE_TIME 先写 0，
            # 后续由求解端场景0生成 PROM_READY/PROM_DUE 并冻结写回 nodes_*_promise.csv
            # =========================
            ready = 0.0
            due = 0.0

            w.writerow({
                "NODE_ID": node_id,
                "NODE_TYPE": "customer",
                "ORIG_X": fmt(x), "ORIG_Y": fmt(y),
                "DEMAND": fmt(dem),
                "READY_TIME": fmt(ready),
                "DUE_TIME": fmt(due),
            })
            node_id += 1


        # 写出客户（先圈内，后圈外）
        for (x, y, dem) in base_customers:
            write_customer(x, y, dem, is_truck=False)
        for (x, y, dem) in truck_customers:
            write_customer(x, y, dem, is_truck=True)

def generate_instance(cfg: GenConfig, out_csv: str):
    random.seed(cfg.seed)

    # 单位换算
    drone_oneway_km = cfg.drone_roundtrip_km / 2.0
    drone_oneway_units = drone_oneway_km * cfg.units_per_km
    truck_speed_units_per_h = cfg.truck_speed_kmh * cfg.units_per_km

    bases, central = generate_fixed_bases_and_central(cfg)

    n_truck = int(round(cfg.n_customers * cfg.truck_customer_ratio))
    n_truck = max(0, min(cfg.n_customers, n_truck))
    n_base = cfg.n_customers - n_truck

    base_clients = generate_base_clients(bases, central, drone_oneway_units, n_base, cfg)
    global_xy = [(x, y) for x, y, _ in base_clients]

    truck_clients = generate_truck_clients_outside_coverage(
        bases, central, drone_oneway_units, n_truck, cfg, global_xy
    )

    # 用于“到达时间特征”的路线
    # 中文注释：用 OR-Tools 求参考路线（用于生成时间窗的到达时刻特征）
        # 中文注释：规模越大，给 OR-Tools 更充足的搜索时间（保证参考路线更合理且可复现）
    time_limit_s = 2 if cfg.n_customers <= 100 else (5 if cfg.n_customers <= 300 else 10)

    route_coords = plan_truck_route_ortools_for_tw(
        bases, central, truck_clients,
        time_limit_s=int(time_limit_s),
        seed=cfg.seed,
        use_local_search=True,
        fallback_to_nn=False
    )

    write_csv(out_csv, bases, central, base_clients, truck_clients, route_coords, cfg, truck_speed_units_per_h)

    # 中文注释：输出复现信息 meta.json（包含参数与生成器指纹），便于论文复现实验
    meta_path = out_csv[:-4] + "_meta.json" if str(out_csv).lower().endswith(".csv") else (str(out_csv) + "_meta.json")
    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "nodes_csv": out_csv,
        "cfg": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        meta["generator_file"] = __file__
        with open(__file__, "rb") as _f:
            meta["generator_sha256"] = hashlib.sha256(_f.read()).hexdigest()
    except Exception as _e:
        meta["generator_file"] = None
        meta["generator_sha256"] = None
        meta["generator_hash_error"] = str(_e)
    # 中文注释：记录 OR-Tools 版本，确保数据集可复现
    try:
        import ortools
        meta["ortools_version"] = ortools.__version__
    except Exception:
        meta["ortools_version"] = None

    with open(meta_path, "w", encoding="utf-8") as _f:
        json.dump(meta, _f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_csv}")
    print(f"[INFO] n_customers={cfg.n_customers}, bases={len(bases)}, base_customers={len(base_clients)}, truck_customers={len(truck_clients)}")
    print(f"[INFO] visual_range={cfg.visual_range:.1f} (axis={cfg.visual_range:.0f}×{cfg.visual_range:.0f}), units_per_km={cfg.units_per_km:.1f}")
    print(f"[INFO] drone_roundtrip_km={cfg.drone_roundtrip_km:.1f} => one_way_units={drone_oneway_units:.1f}")
    print(f"[INFO] truck_road_factor={cfg.truck_road_factor:.2f}（时间窗生成按等效慢速 v_eff=v/road；求解端按 truck_dist=euclid×road）")

# =========================
# 离线 events.csv 生成（固定决策点 + 在线揭示）
# 说明：events.csv 仅提供“事实”（事件发生时刻与新坐标），不写 ACCEPT/REJECT
# =========================

def _read_nodes_csv(nodes_csv: str):
    """读取 nodes.csv，返回 central, bases, customers(list of dict)。
    只依赖字段：NODE_ID,NODE_TYPE,ORIG_X,ORIG_Y
    """
    central = None
    bases = []
    customers = []
    with open(nodes_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nid = int(float(row["NODE_ID"]))
            ntype = str(row["NODE_TYPE"]).strip().lower()
            x = float(row["ORIG_X"])
            y = float(row["ORIG_Y"])
            if ntype == "central":
                central = (x, y, nid)
            elif ntype == "base":
                bases.append((x, y, nid))
            else:
                customers.append({"NODE_ID": nid, "X": x, "Y": y})
    if central is None:
        raise ValueError("nodes.csv 未找到 NODE_TYPE=central")
    return central, bases, customers


def _predict_tau_ref(
        central_xy: Tuple[float, float],
        customers_xy: List[Tuple[int, float, float]],
        *,
        base_points: List[Tuple[float, float, int]],
        r_db_units: float,
        units_per_km: float,
        truck_speed_kmh: float,
        truck_road_factor: float,
        drone_speed_kmh: float,
        drones_per_base: int = 3,
) -> Dict[int, float]:
    """确定性参考预测器 Predictor() —— 方案B（混合快速预测，用于离线 events 护栏筛选）。

    目的：尽量减少“事件在决策点已服务完 -> EXPIRED”的浪费。
    思路：用一个固定、确定性的“粗粒度混合调度”来预测每个客户的参考服务时刻 τ_i^ref，
         决策点 t_k 只挑 τ_i^ref >= t_k + Δlook 的客户做扰动候选。

    预测器包含两部分：
      (1) 卡车侧：只考虑“基地节点 + 圈外客户(不在任何基地覆盖圈)”组成的确定性 NN 路线，
          得到各基地/圈外客户的到达时刻。
      (2) 无人机侧：每个基地 b 在卡车到达时刻 A_b 后才可起飞；
          假设每个基地有 drones_per_base 台无人机，采用贪心并行调度（最早可用机器），
          以 round-trip 时间为作业时长，预测每个客户的送达时刻（取一程到达客户的时间）。

    备注：这是“离线脚本护栏”的预测器，不追求与 ALNS 完全一致；但它是确定的、可复现的，
         且能显著降低因无人机提前完成导致的 EXPIRED。
    """
    if units_per_km <= 0:
        raise ValueError("units_per_km 必须 > 0")
    if truck_speed_kmh <= 0 or drone_speed_kmh <= 0:
        raise ValueError("truck_speed_kmh/drone_speed_kmh 必须 > 0")
    drones_per_base = max(1, int(drones_per_base))

    cx, cy = float(central_xy[0]), float(central_xy[1])
    units_per_km = float(units_per_km)
    v_truck = float(truck_speed_kmh)
    v_drone = float(drone_speed_kmh)
    road = float(truck_road_factor)
    r_units = float(r_db_units)

    def dist_units(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def dist_km(x1: float, y1: float, x2: float, y2: float) -> float:
        return dist_units(x1, y1, x2, y2) / units_per_km

    def truck_time_h_from_units(d_units: float) -> float:
        # 卡车：距离先换算 km 再乘 road_factor
        return (road * (d_units / units_per_km)) / v_truck

    # ---- 1) 客户归属：覆盖基地 -> 最近基地；否则圈外卡车客户 ----
    bases_xy: Dict[int, Tuple[float, float]] = {int(bid): (float(bx), float(by)) for (bx, by, bid) in base_points}

    # 覆盖关系（可能多基地覆盖，但通常只有一个）
    cust_to_base: Dict[int, int] = {}
    truck_only_customers: List[int] = []
    for nid, x, y in customers_xy:
        x = float(x); y = float(y)
        covered = []
        for bid, (bx, by) in bases_xy.items():
            if dist_units(x, y, bx, by) <= r_units + 1e-9:
                covered.append(bid)
        if not covered:
            truck_only_customers.append(int(nid))
        else:
            # 选最近覆盖基地（确定性：距离相同则取 id 小的）
            best = min(covered, key=lambda bid: (dist_units(x, y, bases_xy[bid][0], bases_xy[bid][1]), bid))
            cust_to_base[int(nid)] = int(best)

    # ---- 2) 构造卡车参考路线：访问“有无人机客户的基地 + 圈外客户” ----
    # [回退版] 使用最近邻 (NN) 策略，逻辑简单鲁棒

    # 需要访问的基地（不含 central 本身；central 到达时刻为 0）
    bases_to_visit = sorted({b for b in cust_to_base.values() if b in bases_xy})
    # NN 访问顺序：central -> ... -> central
    unvisited = set(bases_to_visit) | set(truck_only_customers)

    # 将“基地/客户”统一成可查询坐标的节点表
    node_xy: Dict[int, Tuple[float, float]] = {}
    for bid, (bx, by) in bases_xy.items():
        node_xy[int(bid)] = (bx, by)
    for nid, x, y in customers_xy:
        node_xy[int(nid)] = (float(x), float(y))

    # 到达时刻（小时）
    arrive_time: Dict[int, float] = {}
    cur_x, cur_y = cx, cy
    cur_t = 0.0

    while unvisited:
        # NN：选距离最近的未访问点（tie -> id 小）
        nxt = min(unvisited, key=lambda nid: (dist_units(cur_x, cur_y, node_xy[nid][0], node_xy[nid][1]), nid))
        d_units = dist_units(cur_x, cur_y, node_xy[nxt][0], node_xy[nxt][1])
        cur_t += truck_time_h_from_units(d_units)
        arrive_time[int(nxt)] = cur_t
        cur_x, cur_y = node_xy[nxt]
        unvisited.remove(nxt)
    # ---- 3) 无人机并行调度：每个基地 b 从 A_b 时刻开始派送 ----
    tau_ref: Dict[int, float] = {}

    # 圈外客户：卡车到达时刻
    for nid in truck_only_customers:
        tau_ref[int(nid)] = arrive_time.get(int(nid), float("inf"))

    # 基地客户：按 roundtrip 时间 SPT 排序 + 多机最早可用
    # 对每个基地维护 drones_per_base 台无人机的“可用时刻”
    base_to_customers: Dict[int, List[int]] = {}
    for nid, bid in cust_to_base.items():
        base_to_customers.setdefault(int(bid), []).append(int(nid))

    for bid, cust_list in base_to_customers.items():
        bx, by = bases_xy[bid]
        A_b = arrive_time.get(int(bid), 0.0 if (abs(bx - cx) < 1e-9 and abs(by - cy) < 1e-9) else float("inf"))
        if A_b == float("inf"):
            # 理论上不会发生：需要访问的基地应当在 arrive_time 中
            A_b = 0.0

        # 初始化无人机可用时刻
        avail = [A_b for _ in range(drones_per_base)]

        # 作业参数
        jobs = []
        for nid in cust_list:
            x, y = node_xy[nid]
            one_km = dist_km(bx, by, x, y)
            t_one = one_km / v_drone
            t_round = (2.0 * one_km) / v_drone
            jobs.append((t_round, t_one, nid))

        # SPT：roundtrip 从小到大（更接近“尽快送完”）
        jobs.sort(key=lambda z: (z[0], z[2]))

        for t_round, t_one, nid in jobs:
            j = min(range(drones_per_base), key=lambda idx: avail[idx])
            start = avail[j]
            tau_ref[int(nid)] = start + t_one
            avail[j] = start + t_round

    return tau_ref

def _in_cover(x: float, y: float, bx: float, by: float, r_units: float) -> bool:
    return math.hypot(x - bx, y - by) <= r_units + 1e-9

def _assign_home_base(
        x: float, y: float,
        base_points: List[Tuple[float, float, int]],
        r_units: float,
) -> Optional[int]:
    """所属基地：在覆盖该点的基地集合内选最近的基地；无覆盖则 None。"""
    best = None
    best_d = float("inf")
    for bx, by, bid in base_points:
        d = math.hypot(x - bx, y - by)
        if d <= r_units + 1e-9 and d < best_d:
            best_d = d
            best = bid
    return best

def _sample_uniform_in_disk(rng: random.Random, cx: float, cy: float, r: float) -> Tuple[float, float]:
    # 均匀圆盘采样：rad = R*sqrt(u)
    u = rng.random()
    v = rng.random()
    rad = r * math.sqrt(u)
    ang = 2.0 * math.pi * v
    return cx + rad * math.cos(ang), cy + rad * math.sin(ang)

def _sample_out_of_cover(
        rng: random.Random,
        base_points: List[Tuple[float, float, int]],
        r_units: float,
        visual_range: float,
        max_try: int = 2000,
) -> Tuple[float, float]:
    for _ in range(max_try):
        x = rng.random() * visual_range
        y = rng.random() * visual_range
        if all(not _in_cover(x, y, bx, by, r_units) for bx, by, _ in base_points):
            return x, y
    # 兜底：往角落推远（仍尽量出圈）
    return visual_range * 0.98, visual_range * 0.98

def generate_events_csv(
        nodes_csv: str,
        events_csv: str,
        *,
        rho_rel: float = 0.2,
        n_per_dec: int = 25,
        decision_times: List[float] = None,  # 手动指定决策点（小时），如 [1,2,4]；若提供则忽略 n_per_dec 推导
        delta_look_h: float = 0.25,
        delta_avail_min_h: float = 0.25,
        delta_avail_max_h: float = 2.00,
        class_probs: Dict[str, float] = None,
        include_central_as_base: bool = True,
        seed_for_events: int = 0,
        units_per_km: float = 5.0,
        truck_speed_kmh: float = 30.0,
        truck_road_factor: float = 1.5,
        drone_speed_kmh: float = 60.0,
        drones_per_base: int = 3,

        drone_roundtrip_km: float = 10.0,
        visual_range: float = 100.0,
):
    """按定案方案生成 events.csv（事件仅在决策点发生）。
    - 若 decision_times 提供：使用其作为决策点集合 T（可不等间隔）；
    - 否则：默认 T={1..K}，其中 K=ceil(N/n_per_dec)，步长 1h。
    """
    if class_probs is None:
        class_probs = {"IN_DB": 0.6, "CROSS_DB": 0.2, "OUT_DB": 0.2}

    central, bases, customers = _read_nodes_csv(nodes_csv)

    # 基地集合：B 或 B∪{central}
    base_points = bases.copy()
    if include_central_as_base:
        base_points.append((central[0], central[1], int(central[2])))

    # 覆盖半径（单位：坐标单位）
    r_db_units = (float(drone_roundtrip_km) / 2.0) * float(units_per_km)
    # 决策点集合 T：手动指定优先，否则默认 T={1..K}（步长 1h）
    N = len(customers)
    if decision_times is not None and len(decision_times) > 0:
        # 手动决策点：允许非等间隔；去重、排序，并统一为 float（小时）
        T_list = sorted({float(t) for t in decision_times})
        if len(T_list) == 0:
            raise ValueError('decision_times 为空')
        # 可选：限制为非负时刻（按需调整）
        if any(t < 0 for t in T_list):
            raise ValueError('decision_times 中存在负时间')
        K = len(T_list)
    else:
        K = max(1, int(math.ceil(float(N) / float(n_per_dec))))
        T_list = list(range(1, K + 1))

    # 总扰动数与各决策点分配
    N_rel = int(math.floor(float(rho_rel) * float(N)))
    base_cnt = N_rel // K
    rem = N_rel % K
    M_k = {t: base_cnt + (1 if (idx < rem) else 0) for idx, t in enumerate(T_list)}

    # Predictor(): 计算 τ_i^ref
    customers_xy = [(int(c["NODE_ID"]), float(c["X"]), float(c["Y"])) for c in customers]
    tau_ref = _predict_tau_ref(
        (central[0], central[1]),
        customers_xy,
        base_points=base_points,
        r_db_units=r_db_units,
        units_per_km=units_per_km,
        truck_speed_kmh=truck_speed_kmh,
        truck_road_factor=truck_road_factor,
        drone_speed_kmh=drone_speed_kmh,
        drones_per_base=drones_per_base,
    )

    # Predictor 统计（用于自检：τ_ref 是否覆盖到 1~K 小时范围）
    _vals = list(tau_ref.values())
    if _vals:
        _mn = min(_vals); _mx = max(_vals); _avg = sum(_vals) / max(1, len(_vals))
        print(f"[PRED] tau_ref(B) h min/mean/max = {_mn:.2f}/{_avg:.2f}/{_mx:.2f}")

    # ====== 事件生成：按决策点分配数量，并尽量按比例生成三类事件 ======
    # 事件随机源：与 nodes 生成随机性解耦（与 nodes 随机性解耦）
    rng = random.Random(int(seed_for_events))

    truck_road_factor = float(truck_road_factor)
    if truck_road_factor <= 0:
        raise ValueError('truck_road_factor 必须为正')

    # 类别顺序固定（保证可复现）
    classes = ["IN_DB", "CROSS_DB", "OUT_DB"]
    weights = [float(class_probs.get(c, 0.0)) for c in classes]
    wsum = sum(weights)
    if wsum <= 1e-12:
        raise ValueError("class_probs 权重之和为 0")
    weights = [w / wsum for w in weights]

    def _alloc_counts(total: int) -> Dict[str, int]:
        """把 total 按 weights 分配到三个类别，采用 floor+最大余数法，保证整数且总和=total。"""
        exp = [total * w for w in weights]
        base = [int(math.floor(v)) for v in exp]
        rem = total - sum(base)
        frac = [exp[i] - base[i] for i in range(len(base))]
        # 稳定排序：余数大优先；余数相同按类顺序
        order = sorted(range(len(base)), key=lambda i: (-frac[i], i))
        for i in order[:rem]:
            base[i] += 1
        return {classes[i]: base[i] for i in range(len(classes))}

    # 打印关键元信息，便于自检
    print(f"[EVENTS] decision_times={decision_times} -> T_list={T_list}  K={K}")
    print(f"[EVENTS] N={N}, rho_rel={rho_rel} => N_rel={N_rel}, M_k={M_k}")
    print(f"[EVENTS] class_probs(norm)=" +
          ",".join([f"{c}={weights[i]:.3f}" for i, c in enumerate(classes)]))

    # 每个客户最多扰动一次
    used = set()

    events = []
    event_id = 1

    # 统计：期望类别 vs 实际类别（含降级）
        # 事件类别的“比例”在样本数很小时会出现整数化误差（例如 total=2 时 0.2 往往被舍入为 0）。
    # 为了让 class_probs 更符合“长期期望比例”的语义：先在全局对 N_rel 做一次整数分配，
    # 再把全局各类数量拆分到每个决策时刻（同时满足每个时刻的事件总数 M_k）。
    global_want = _alloc_counts(int(N_rel))

    def _alloc_global_to_times() -> Dict[float, Dict[str, int]]:
        """把 global_want 拆分到每个决策时刻。
        约束：
          (1) 对任意 t：sum_c want[t][c] = M_k[t]
          (2) 对任意 c：sum_t want[t][c] = global_want[c]
        规则：按 weights 的期望值做 floor+最大余数，同时受“剩余额度”约束；剩余空位按剩余额度补齐。
        """
        rem = {c: int(global_want[c]) for c in classes}
        want_by_t: Dict[float, Dict[str, int]] = {}

        for t_k in T_list:
            need = int(M_k[t_k])
            if need <= 0:
                want_by_t[t_k] = {c: 0 for c in classes}
                continue

            exp = [need * w for w in weights]
            base = [min(int(math.floor(exp[i])), rem[classes[i]]) for i in range(len(classes))]
            slots = need - sum(base)

            frac = [exp[i] - math.floor(exp[i]) for i in range(len(classes))]
            order = sorted(range(len(classes)), key=lambda i: (-frac[i], i))  # 稳定：余数大优先；并列按类顺序

            for i in order:
                if slots <= 0:
                    break
                c = classes[i]
                if rem[c] > base[i]:
                    base[i] += 1
                    slots -= 1

            # 若仍有空位（某些类别额度已用尽），则按剩余额度从大到小补齐
            while slots > 0:
                order2 = sorted(range(len(classes)), key=lambda i: (-(rem[classes[i]] - base[i]), i))
                placed = False
                for i in order2:
                    c = classes[i]
                    if rem[c] - base[i] > 0:
                        base[i] += 1
                        slots -= 1
                        placed = True
                        break
                if not placed:
                    break

            want_by_t[t_k] = {classes[i]: int(base[i]) for i in range(len(classes))}
            for i in range(len(classes)):
                rem[classes[i]] -= base[i]
        return want_by_t

    want_plan = _alloc_global_to_times()

    # 自检：行/列守恒；若失败则回退到“逐时刻分配”（极少发生，主要在 N_rel 很小且 K 很大时）
    _chk_cols = {c: sum(want_plan[t][c] for t in T_list) for c in classes}
    _chk_rows = {t: sum(want_plan[t][c] for c in classes) for t in T_list}
    if any(_chk_cols[c] != global_want[c] for c in classes) or any(_chk_rows[t] != int(M_k[t]) for t in T_list):
        print("[EVENTS-WARN] 类别全局拆分未通过守恒自检，回退为逐决策点分配（可能出现 OUT_DB=0）。")
        want_plan = {t: _alloc_counts(int(M_k[t])) for t in T_list}

    stat_desired = {t: {c: int(want_plan[t].get(c, 0)) for c in classes} for t in T_list}
    stat_real = {t: {c: 0 for c in classes} for t in T_list}
    # ====== [新增] 预计算参考路径的拓扑序列 (Sequence Rank) ======
    print("[EVENTS] Generating reference topology using OR-Tools (for sequence ranking)...")

    # 1. 准备 OR-Tools 输入：只包含 central + 所有客户 (bases 不参与排序，因为它们是必经点)
    # Ref: central is index 0
    ref_nodes_input = [(central[0], central[1])]
    ref_ids_map = {0: -1}  # OR-Tools index -> Real Node ID (-1 for central)

    current_idx = 1
    # 加入所有客户（不管圈内圈外，统统排个序）
    for c in customers:
        ref_nodes_input.append((float(c["X"]), float(c["Y"])))
        ref_ids_map[current_idx] = int(c["NODE_ID"])
        current_idx += 1

    # 调用 OR-Tools (借用现有的 plan_truck_route_ortools_for_tw 函数)
    # 把所有客户都塞到 truck_clients 参数里，bases 传空
    ref_route_coords = plan_truck_route_ortools_for_tw(
        bases=[],
        central=(central[0], central[1]),
        truck_clients=[(x, y, 0) for x, y in ref_nodes_input[1:]],  # 跳过central
        time_limit_s=5,  # 给多一点时间保证质量
        seed=seed_for_events,
        use_local_search=True
    )

    # 2. 将坐标序列转换为 ID 的进度表 (0.0 ~ 1.0)
    # ref_route_coords 是有序的坐标列表 [(x,y), (x,y)...]
    node_progress = {}  # {NODE_ID: progress_float}
    total_steps = len(ref_route_coords)

    # 建立坐标指纹到 ID 的映射（防止浮点误差）
    def get_coord_key(x, y):
        return (round(x, 4), round(y, 4))

    coord_to_id = {}
    for c in customers:
        coord_to_id[get_coord_key(float(c['X']), float(c['Y']))] = int(c['NODE_ID'])

    for idx, (rx, ry) in enumerate(ref_route_coords):
        key = get_coord_key(rx, ry)
        if key in coord_to_id:
            nid = coord_to_id[key]
            # 记录该客户在路径中的相对位置 (0.0 = 起点, 1.0 = 终点)
            if nid not in node_progress:
                node_progress[nid] = idx / max(1, total_steps - 1)

    print(f"[EVENTS] Sequence ranking built for {len(node_progress)} customers.")
    # =================================================================
    for t_k in T_list:

        need = int(M_k[t_k])
        if need <= 0:
            continue
        # ====== [修改] 基于序列进度的候选筛选 ======

        # 1. 计算当前时间进度 (0.0 ~ 1.0)
        # 假设 decision_times 是有序的，用最后一个时间作为结束基准
        # max_t 用于归一化：比如当前是第 4 小时，总共 8 小时，进度就是 0.5
        max_t = float(T_list[-1])
        time_progress = float(t_k) / max_t

        # 2. 确定允许的进度窗口
        # 核心逻辑：如果在 50% 的时间点发请求，那么该客户在路径中的位置最好 > 50%
        # 引入一个安全缓冲 (buffer)，比如 0.05 (5%)
        # 含义：t=4(50%)时，只选排名在 55% 之后的客户
        min_seq_progress = time_progress + 0.05

        # 3. 筛选候选人
        cand = []
        for c in customers:
            nid = int(c["NODE_ID"])
            if nid in used:
                continue

            # 获取该客户在参考路径中的位置 (如果在 map 里没找到，默认给 1.0 视为最晚)
            seq_pos = node_progress.get(nid, 1.0)

            # 判据：客户排在当前时间点之后，说明“还没被访问”
            if seq_pos >= min_seq_progress:
                cand.append(c)

        # 兜底 A：如果选不出人（比如到了最后时刻），尝试稍微放宽
        if len(cand) < need:
            # 放宽：只要求排在后 20% 的人（不管当前几点了，只要是最后那批就行）
            cand = [c for c in customers
                    if int(c["NODE_ID"]) not in used
                    and node_progress.get(int(c["NODE_ID"]), 0) > 0.8]

        # 兜底 B：还不行就全量
        if len(cand) < need:
            cand = [c for c in customers if int(c["NODE_ID"]) not in used]
            print(f"[WARN] t={t_k}: Sequence buffer exhausted, using fallback.")

        # 打乱顺序，避免每次都选同一个，保证随机性
        rng.shuffle(cand)
        # ============================================

        # ——关键：先按比例确定“本决策点各类事件数量”，再从可行客户池中抽取——
        want = stat_desired[t_k]
# 预计算每个候选客户的 home（所属DB；若不在任何DB覆盖圈内则为 None）
        cand_info = []
        for c in cand:
            nid = int(c["NODE_ID"])
            ox, oy = float(c["X"]), float(c["Y"])
            home = _assign_home_base(ox, oy, base_points, r_db_units)
            has_other = False
            if home is not None:
                has_other = any(p[2] != home for p in base_points)
            cand_info.append((nid, ox, oy, home, has_other))

        # 按类别构造可行客户池（不改变“事件事实”定义）
        pool_in = [x for x in cand_info if x[3] is not None]                # home!=None
        pool_cross = [x for x in cand_info if x[3] is not None and x[4]]    # home!=None 且存在其他DB
        pool_out = cand_info[:]                                             # 总是可行

        def pick_from_pool(pool, k_take):
            take = []
            if k_take <= 0:
                return take
            for item in pool:
                nid = item[0]
                if nid in used:
                    continue
                used.add(nid)
                take.append(item)
                if len(take) >= k_take:
                    break
            return take

        # 先满足 IN_DB、CROSS_DB，再把剩余给 OUT_DB
        take_in = pick_from_pool(pool_in, want["IN_DB"])
        take_cross = pick_from_pool(pool_cross, want["CROSS_DB"])
        remain = need - (len(take_in) + len(take_cross))
        take_out = pick_from_pool(pool_out, remain)

        # 生成事件
        for nid, ox, oy, home, has_other in take_in:
            evt_class = "IN_DB"
            bx, by, _ = next(p for p in base_points if p[2] == home)
            nx, ny = _sample_uniform_in_disk(rng, bx, by, r_db_units)
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid,
                           "NEW_X": float(nx), "NEW_Y": float(ny), "EVENT_CLASS": evt_class,
                           "DELTA_AVAIL_H": float(rng.uniform(delta_avail_min_h, delta_avail_max_h))})
            stat_real[t_k][evt_class] += 1
            event_id += 1

        for nid, ox, oy, home, has_other in take_cross:
            evt_class = "CROSS_DB"
            others = [p for p in base_points if p[2] != home]
            bx, by, _ = rng.choice(others)
            nx, ny = _sample_uniform_in_disk(rng, bx, by, r_db_units)
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid,
                           "NEW_X": float(nx), "NEW_Y": float(ny), "EVENT_CLASS": evt_class,
                           "DELTA_AVAIL_H": float(rng.uniform(delta_avail_min_h, delta_avail_max_h))})
            stat_real[t_k][evt_class] += 1
            event_id += 1

        for nid, ox, oy, home, has_other in take_out:
            evt_class = "OUT_DB"
            nx, ny = _sample_out_of_cover(rng, base_points, r_db_units, float(visual_range))
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid,
                           "NEW_X": float(nx), "NEW_Y": float(ny), "EVENT_CLASS": evt_class,
                           "DELTA_AVAIL_H": float(rng.uniform(delta_avail_min_h, delta_avail_max_h))})
            stat_real[t_k][evt_class] += 1
            event_id += 1

    # 排序
    events.sort(key=lambda e: (e["EVENT_TIME"], e["EVENT_ID"]))

    # 打印统计，便于核对“时间分配/比例/降级”
    print("[EVENTS] per-time desired vs real counts:")
    for t in T_list:
        d = stat_desired[t]
        r = stat_real[t]
        print(f"  t={t}: desired(IN={d['IN_DB']},CROSS={d['CROSS_DB']},OUT={d['OUT_DB']})  "
              f"real(IN={r['IN_DB']},CROSS={r['CROSS_DB']},OUT={r['OUT_DB']})")
    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EVENT_ID", "EVENT_TIME", "NODE_ID", "NEW_X", "NEW_Y", "EVENT_CLASS", "DELTA_AVAIL_H"])
        w.writeheader()
        for e in events:
            w.writerow({
                "EVENT_ID": int(e["EVENT_ID"]),
                "EVENT_TIME": f"{float(e['EVENT_TIME']):.2f}",
                "NODE_ID": int(e["NODE_ID"]),
                "NEW_X": f"{float(e['NEW_X']):.3f}",
                "NEW_Y": f"{float(e['NEW_Y']):.3f}",
                "EVENT_CLASS": str(e["EVENT_CLASS"]),
                "DELTA_AVAIL_H": f"{float(e.get('DELTA_AVAIL_H', 0.0)):.2f}",
            })


def generate_nodes_and_events(
        cfg: GenConfig,
        out_nodes_csv: str,
        out_events_csv: str,
        *,
        rho_rel: float = 0.2,
        n_per_dec: int = 25,
        decision_times: List[float] = None,  # 手动指定决策点（小时），如 [1,2,4]；若提供则忽略 n_per_dec 推导
        delta_look_h: float = 0.25,
        delta_avail_min_h: float = 0.25,
        delta_avail_max_h: float = 2.00,
        class_probs: Dict[str, float] = None,
        include_central_as_base: bool = True,
        truck_road_factor: float = 1.5,
):
    """生成 nodes.csv + events.csv（离线脚本），events 的随机性与 nodes 解耦。"""
    # 统一路况系数：用于 nodes 时间窗 + events 预测筛选（与求解端一致）
    cfg.truck_road_factor = float(truck_road_factor)

    generate_instance(cfg, out_nodes_csv)
    seed_for_events = int(cfg.seed) + 1000003

    generate_events_csv(
        out_nodes_csv,
        out_events_csv,
        rho_rel=rho_rel,
        n_per_dec=n_per_dec,
        decision_times=decision_times,
        delta_look_h=delta_look_h,
        class_probs=class_probs,
        include_central_as_base=include_central_as_base,
        truck_road_factor=float(cfg.truck_road_factor),
        seed_for_events=seed_for_events,
        units_per_km=float(cfg.units_per_km),
        truck_speed_kmh=float(cfg.truck_speed_kmh),
        drone_speed_kmh=float(cfg.drone_speed_kmh),
        drones_per_base=int(cfg.drones_per_base),
        drone_roundtrip_km=float(cfg.drone_roundtrip_km),
        visual_range=float(cfg.visual_range),
    )

    # 中文注释：events 同步输出 meta.json，记录事件生成参数（便于论文复现）
    events_meta_path = out_events_csv[:-4] + "_meta.json" if str(out_events_csv).lower().endswith(".csv") else (str(out_events_csv) + "_meta.json")
    events_meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "nodes_csv": out_nodes_csv,
        "events_csv": out_events_csv,
        "seed_for_events": seed_for_events,
        "event_params": {
            "rho_rel": rho_rel,
            "n_per_dec": n_per_dec,
            "decision_times": decision_times,
            "delta_look_h": delta_look_h,
            "class_probs": class_probs,
            "include_central_as_base": include_central_as_base,
            "truck_road_factor": float(cfg.truck_road_factor),
        },
        "cfg": asdict(cfg),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        events_meta["generator_file"] = __file__
        with open(__file__, "rb") as _f:
            events_meta["generator_sha256"] = hashlib.sha256(_f.read()).hexdigest()
    except Exception as _e:
        events_meta["generator_file"] = None
        events_meta["generator_sha256"] = None
        events_meta["generator_hash_error"] = str(_e)

    with open(events_meta_path, "w", encoding="utf-8") as _f:
        json.dump(events_meta, _f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 默认：生成 25 客户、5 个 seed 的离线 nodes.csv + events.csv
    seed_list = [2023]
    # seed_list = [2021, 2022, 2023, 2024, 2025]
    n_customers = 200

    # 事件参数（按论文实验统一）
    rho_rel = 0.2
    n_per_dec = 25
    # 手动指定决策点（小时），可不等间隔；例如 [1, 2] 或 [0.5, 1.5, 3]
    # decision_times = [1, 2]
    # decision_times = [1, 2, 3]
    # decision_times = [1, 2, 3, 4, 5, 6]
    decision_times = [1, 2, 3, 4, 5, 6, 7, 8]
    # 注意：如果 decision_times 非空，则生成 events 时将忽略 n_per_dec 推导 K
    delta_look_h = 6
    class_probs = {"IN_DB": 0.6, "CROSS_DB": 0.2, "OUT_DB": 0.2}
    include_central_as_base = True

    for s in seed_list:
        cfg = GenConfig(
            n_customers=n_customers,visual_range=200.0,
            drone_roundtrip_km=10.0,
            truck_road_factor=1.5,
            seed=int(s), base_count_override={25: 3,
                50: 6,
                100: 8, 200: 12},
        )
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_nodes = f"nodes_{cfg.n_customers}_seed{cfg.seed}_{now}.csv"
        out_events = f"events_{cfg.n_customers}_seed{cfg.seed}_{now}.csv"
        generate_nodes_and_events(
            cfg, out_nodes, out_events,
            rho_rel=rho_rel,
            n_per_dec=n_per_dec,
            decision_times=decision_times,
            delta_look_h=delta_look_h,
            class_probs=class_probs,
            include_central_as_base=include_central_as_base,
        )
