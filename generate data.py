# -*- coding: utf-8 -*-
"""
动态卡车-无人机协同配送 (VRPTW-CL) 基准数据集生成器
=============================================================================
设计目标：
1. 生成静态客户与基站拓扑 (nodes.csv)，时间窗将在求解端初始场景(Scene 0)中生成并冻结。
2. 生成动态位置变更请求 (events.csv)，严格按照设定的概率分布生成各类事件。
3. 采用分层结构化输出：datasets_promise/{scale}_data/{seed}/...
=============================================================================
"""

import csv
import os
import math
import random
import datetime
import json
import sys
import platform
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional


# =========================
# 基础工具与数据类
# =========================
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_in_annulus(cx: float, cy: float, r_min: float, r_max: float) -> Tuple[float, float]:
    theta = random.random() * 2.0 * math.pi
    r = r_min + math.sqrt(random.random()) * (r_max - r_min)
    return cx + r * math.cos(theta), cy + r * math.sin(theta)


def fmt(v):
    if v is None: return ""
    if isinstance(v, (int, float)): return f"{float(v):.2f}"
    return str(v)


@dataclass
class GenConfig:
    n_customers: int = 100
    visual_range: float = 100.0
    units_per_km: float = 5.0

    truck_speed_kmh: float = 30.0
    truck_road_factor: float = 1.5
    drone_speed_kmh: float = 60.0
    drones_per_base: int = 3
    drone_roundtrip_km: float = 10.0

    base_ratio: float = 0.12
    min_bases: int = 1
    base_count_override: dict = None
    truck_customer_ratio: float = 0.20

    min_dist_global: float = 7.0
    min_dist_within_ring: float = 10.0
    ring_min_ratio: float = 0.25
    ring_max_ratio: float = 0.90

    drone_small_prob: float = 0.80
    drone_small_range: Tuple[int, int] = (1, 3)
    drone_large_range: Tuple[int, int] = (4, 5)
    truck_mid_prob: float = 0.70
    truck_mid_range: Tuple[int, int] = (6, 8)
    truck_large_range: Tuple[int, int] = (9, 10)

    center_slack_max: float = 0.25
    window_width: float = 0.30
    tw_width_scale_per_100: float = 0.5
    tw_global_shift_ratio: float = 0.15
    truck_due_plus_h: float = 1.0
    drone_due_plus_h: float = 2.0

    seed: int = 2


# =========================
# 设施与需求点生成
# =========================
def generate_fixed_bases_and_central(cfg: GenConfig) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    central = (cfg.visual_range / 2.0, cfg.visual_range / 2.0)
    k = max(cfg.min_bases, int(round(cfg.n_customers * cfg.base_ratio)))
    if cfg.base_count_override and (cfg.n_customers in cfg.base_count_override):
        k = int(cfg.base_count_override[cfg.n_customers])

    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))
    margin = cfg.visual_range * 0.15

    xs = [margin + i * (cfg.visual_range - 2 * margin) / (cols - 1 if cols > 1 else 1) for i in range(cols)]
    ys = [margin + j * (cfg.visual_range - 2 * margin) / (rows - 1 if rows > 1 else 1) for j in range(rows)]

    bases = []
    for y in ys:
        for x in xs:
            if len(bases) >= k: break
            p = (float(x), float(y))
            if distance(p, central) > 1e-6:
                bases.append(p)
        if len(bases) >= k: break
    return bases, central


def sample_drone_demand(cfg: GenConfig) -> int:
    return random.randint(*cfg.drone_small_range) if random.random() < cfg.drone_small_prob else random.randint(
        *cfg.drone_large_range)


def sample_truck_demand(cfg: GenConfig) -> int:
    return random.randint(*cfg.truck_mid_range) if random.random() < cfg.truck_mid_prob else random.randint(
        *cfg.truck_large_range)


def generate_base_clients(bases, central, drone_oneway_units, n_base_customers, cfg) -> List[Tuple[float, float, int]]:
    facilities = bases + [central]
    r_min = drone_oneway_units * cfg.ring_min_ratio
    r_max = drone_oneway_units * cfg.ring_max_ratio
    clients, global_xy = [], []
    per, extra = n_base_customers // len(facilities), n_base_customers % len(facilities)

    for j, (fx, fy) in enumerate(facilities):
        need = per + (1 if j < extra else 0)
        placed_local = []
        for _ in range(need):
            last_xy = (fx, fy)
            for _ in range(300):
                x, y = sample_in_annulus(fx, fy, r_min, r_max)
                x, y = clamp(x, 0.0, cfg.visual_range), clamp(y, 0.0, cfg.visual_range)
                last_xy = (x, y)
                if any(distance((x, y), p) < cfg.min_dist_within_ring for p in placed_local): continue
                if any(distance((x, y), p) < cfg.min_dist_global for p in global_xy): continue
                break
            x, y = last_xy
            clients.append((x, y, sample_drone_demand(cfg)))
            placed_local.append((x, y))
            global_xy.append((x, y))
    return clients


def generate_truck_clients_outside_coverage(bases, central, drone_oneway_units, n_truck_customers, cfg, existing_xy) -> \
List[Tuple[float, float, int]]:
    facilities = bases + [central]
    truck_clients = []
    tries, max_tries = 0, 200000

    while len(truck_clients) < n_truck_customers and tries < max_tries:
        tries += 1
        x, y = random.uniform(0.0, cfg.visual_range), random.uniform(0.0, cfg.visual_range)
        if any(distance((x, y), f) <= drone_oneway_units for f in facilities): continue
        if any(distance((x, y), p) < cfg.min_dist_global for p in existing_xy): continue
        if any(distance((x, y), (tx, ty)) < cfg.min_dist_global for tx, ty, _ in truck_clients): continue

        truck_clients.append((x, y, sample_truck_demand(cfg)))
        existing_xy.append((x, y))
    return truck_clients


# =========================
# OR-Tools TSP 参考路径生成
# =========================
def plan_truck_route_nearest_neighbor(bases, central, truck_clients):
    nodes = [central] + bases + [(x, y) for x, y, _ in truck_clients]
    visited, route_idx, cur = [False] * len(nodes), [0], 0
    visited[0] = True
    while len(route_idx) < len(nodes):
        nxt, best = None, float("inf")
        for j in range(len(nodes)):
            if not visited[j]:
                d = distance(nodes[cur], nodes[j])
                if d < best: best, nxt = d, j
        visited[nxt] = True
        route_idx.append(nxt)
        cur = nxt
    route_idx.append(0)
    return [nodes[i] for i in route_idx]


def plan_truck_route_ortools_for_tw(bases, central, truck_clients, time_limit_s=2, seed=1, use_local_search=True,
                                    fallback_to_nn=False, dist_scale=1000):
    nodes = [central] + list(bases) + [(x, y) for x, y, _ in truck_clients]
    n = len(customers)
    if n <= 2: return [central, central]

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as e:
        if fallback_to_nn: return plan_truck_route_nearest_neighbor(bases, central, truck_clients)
        raise RuntimeError(f"[TW-ROUTE] OR-Tools 未安装: {e}") from e

    def int_cost(i, j):
        return int(max(1, round(distance(nodes[i], nodes[j]) * dist_scale)))

    dist_mat = [[0 if i == j else int_cost(i, j) for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_cb_idx = routing.RegisterTransitCallback(
        lambda f, t: dist_mat[manager.IndexToNode(f)][manager.IndexToNode(t)])
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if use_local_search:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    params.time_limit.FromSeconds(int(time_limit_s))
    # 【修复】：彻底删除 params.random_seed 避免高版本 OR-Tools 报错

    sol = routing.SolveWithParameters(params)
    if sol is None:
        if fallback_to_nn: return plan_truck_route_nearest_neighbor(bases, central, truck_clients)
        raise RuntimeError("[TW-ROUTE] OR-Tools 未找到可行解。")

    order, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(manager.IndexToNode(idx))
    return [nodes[k] for k in order]


def calculate_arrival_times(route_coords, truck_speed_units_per_h) -> Dict[Tuple[float, float], float]:
    arrival = {route_coords[0]: 0.0}
    t = 0.0
    for i in range(len(route_coords) - 1):
        a, b = route_coords[i], route_coords[i + 1]
        t += distance(a, b) / truck_speed_units_per_h
        arrival[b] = min(arrival.get(b, float('inf')), t)
    return arrival


def nearest_facility(pos, bases, central) -> Tuple[float, float]:
    return min(bases + [central], key=lambda f: distance(pos, f))


# =========================
# nodes.csv 输出
# =========================
def write_csv(csv_file, bases, central, base_customers, truck_customers, route_coords, cfg, truck_speed_units_per_h):
    fieldnames = ["NODE_ID", "NODE_TYPE", "ORIG_X", "ORIG_Y", "DEMAND", "READY_TIME", "DUE_TIME"]
    truck_speed_units_per_h_eff = float(truck_speed_units_per_h) / max(1e-9, float(cfg.truck_road_factor))
    arrival_times = calculate_arrival_times(route_coords, truck_speed_units_per_h_eff)

    node_id = 0
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        w.writerow({"NODE_ID": node_id, "NODE_TYPE": "central", "ORIG_X": fmt(central[0]), "ORIG_Y": fmt(central[1]),
                    "DEMAND": fmt(0.0), "READY_TIME": fmt(0.0), "DUE_TIME": fmt(0.0)})
        node_id += 1
        for b in bases:
            t_b = arrival_times.get(b, 0.0)
            w.writerow(
                {"NODE_ID": node_id, "NODE_TYPE": "base", "ORIG_X": fmt(b[0]), "ORIG_Y": fmt(b[1]), "DEMAND": fmt(0.0),
                 "READY_TIME": fmt(t_b), "DUE_TIME": fmt(t_b)})
            node_id += 1

        for (x, y, dem) in base_customers + truck_customers:
            w.writerow(
                {"NODE_ID": node_id, "NODE_TYPE": "customer", "ORIG_X": fmt(x), "ORIG_Y": fmt(y), "DEMAND": fmt(dem),
                 "READY_TIME": fmt(0.0), "DUE_TIME": fmt(0.0)})
            node_id += 1


def generate_instance(cfg: GenConfig, out_csv: str):
    random.seed(cfg.seed)
    drone_oneway_units = (cfg.drone_roundtrip_km / 2.0) * cfg.units_per_km
    truck_speed_units_per_h = cfg.truck_speed_kmh * cfg.units_per_km

    bases, central = generate_fixed_bases_and_central(cfg)
    n_truck = max(0, min(cfg.n_customers, int(round(cfg.n_customers * cfg.truck_customer_ratio))))
    n_base = cfg.n_customers - n_truck

    base_clients = generate_base_clients(bases, central, drone_oneway_units, n_base, cfg)
    truck_clients = generate_truck_clients_outside_coverage(bases, central, drone_oneway_units, n_truck, cfg,
                                                            [(x, y) for x, y, _ in base_clients])

    time_limit_s = 2 if cfg.n_customers <= 100 else (5 if cfg.n_customers <= 300 else 10)
    route_coords = plan_truck_route_ortools_for_tw(bases, central, truck_clients, time_limit_s=time_limit_s,
                                                   seed=cfg.seed)

    write_csv(out_csv, bases, central, base_clients, truck_clients, route_coords, cfg, truck_speed_units_per_h)


# =========================
# 动态离线 events.csv 生成
# =========================
def _read_nodes_csv(nodes_csv: str):
    central, bases, customers = None, [], []
    with open(nodes_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nid, ntype, x, y = int(float(row["NODE_ID"])), str(row["NODE_TYPE"]).strip().lower(), float(
                row["ORIG_X"]), float(row["ORIG_Y"])
            if ntype == "central":
                central = (x, y, nid)
            elif ntype == "base":
                bases.append((x, y, nid))
            else:
                customers.append({"NODE_ID": nid, "X": x, "Y": y})
    return central, bases, customers


def _predict_tau_ref(central_xy, customers_xy, base_points, r_db_units, units_per_km, truck_speed_kmh,
                     truck_road_factor, drone_speed_kmh, drones_per_base):
    v_truck, v_drone, road = float(truck_speed_kmh), float(drone_speed_kmh), float(truck_road_factor)

    def dist_units(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    bases_xy = {int(bid): (float(bx), float(by)) for bx, by, bid in base_points}
    cust_to_base, truck_only_customers = {}, []

    for nid, x, y in customers_xy:
        covered = [bid for bid, (bx, by) in bases_xy.items() if dist_units(x, y, bx, by) <= r_db_units + 1e-9]
        if not covered:
            truck_only_customers.append(int(nid))
        else:
            best = min(covered, key=lambda bid: (dist_units(x, y, bases_xy[bid][0], bases_xy[bid][1]), bid))
            cust_to_base[int(nid)] = int(best)

    bases_to_visit = sorted({b for b in cust_to_base.values() if b in bases_xy})
    unvisited = set(bases_to_visit) | set(truck_only_customers)
    node_xy = {**bases_xy, **{int(n): (float(x), float(y)) for n, x, y in customers_xy}}

    arrive_time = {}
    cur_x, cur_y, cur_t = central_xy[0], central_xy[1], 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda nid: (dist_units(cur_x, cur_y, node_xy[nid][0], node_xy[nid][1]), nid))
        cur_t += (road * (dist_units(cur_x, cur_y, node_xy[nxt][0], node_xy[nxt][1]) / units_per_km)) / v_truck
        arrive_time[int(nxt)] = cur_t
        cur_x, cur_y = node_xy[nxt]
        unvisited.remove(nxt)

    tau_ref = {nid: arrive_time.get(nid, float("inf")) for nid in truck_only_customers}
    base_to_customers = {}
    for nid, bid in cust_to_base.items(): base_to_customers.setdefault(int(bid), []).append(int(nid))

    for bid, cust_list in base_to_customers.items():
        bx, by = bases_xy[bid]
        A_b = arrive_time.get(int(bid),
                              0.0 if (abs(bx - central_xy[0]) < 1e-9 and abs(by - central_xy[1]) < 1e-9) else float(
                                  "inf"))
        avail = [A_b for _ in range(drones_per_base)]
        jobs = []
        for nid in cust_list:
            x, y = node_xy[nid]
            one_km = dist_units(bx, by, x, y) / units_per_km
            jobs.append(((2.0 * one_km) / v_drone, one_km / v_drone, nid))
        jobs.sort(key=lambda z: (z[0], z[2]))
        for t_round, t_one, nid in jobs:
            j = min(range(drones_per_base), key=lambda idx: avail[idx])
            tau_ref[int(nid)] = avail[j] + t_one
            avail[j] += t_round

    return tau_ref, arrive_time


def _in_cover(x, y, bx, by, r_units): return math.hypot(x - bx, y - by) <= r_units + 1e-9


def _assign_home_base(x, y, base_points, r_units):
    best, best_d = None, float("inf")
    for bx, by, bid in base_points:
        d = math.hypot(x - bx, y - by)
        if d <= r_units + 1e-9 and d < best_d:
            best_d, best = d, bid
    return best


def _sample_uniform_in_disk(rng, cx, cy, r):
    rad, ang = r * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
    return cx + rad * math.cos(ang), cy + rad * math.sin(ang)


def _sample_out_of_cover(rng, base_points, r_units, visual_range):
    for _ in range(2000):
        x, y = rng.random() * visual_range, rng.random() * visual_range
        if all(not _in_cover(x, y, bx, by, r_units) for bx, by, _ in base_points): return x, y
    return visual_range * 0.98, visual_range * 0.98


def generate_events_csv(nodes_csv, events_csv, *, rho_rel, decision_times, delta_look_h, delta_avail_min_h=0.25,
                        delta_avail_max_h=2.00, class_probs=None, include_central_as_base=True, seed_for_events=0,
                        units_per_km=5.0, truck_speed_kmh=30.0, truck_road_factor=1.5, drone_speed_kmh=60.0,
                        drones_per_base=3, drone_roundtrip_km=10.0, visual_range=100.0):
    if class_probs is None: class_probs = {"IN_DB": 0.5, "CROSS_DB": 0.3, "OUT_DB": 0.2}
    central, bases, customers = _read_nodes_csv(nodes_csv)
    base_points = bases.copy()
    if include_central_as_base: base_points.append((central[0], central[1], int(central[2])))

    r_db_units = (float(drone_roundtrip_km) / 2.0) * float(units_per_km)
    N = len(customers)

    T_list = sorted({float(t) for t in decision_times})
    K = len(T_list)

    N_rel = int(math.floor(float(rho_rel) * float(N)))
    M_k = {t: N_rel // K + (1 if idx < N_rel % K else 0) for idx, t in enumerate(T_list)}

    tau_ref, arrive_time = _predict_tau_ref((central[0], central[1]),
                                            [(int(c["NODE_ID"]), float(c["X"]), float(c["Y"])) for c in customers],
                                            base_points=base_points, r_db_units=r_db_units, units_per_km=units_per_km,
                                            truck_speed_kmh=truck_speed_kmh, truck_road_factor=truck_road_factor,
                                            drone_speed_kmh=drone_speed_kmh, drones_per_base=drones_per_base)

    rng = random.Random(int(seed_for_events))
    classes = ["IN_DB", "CROSS_DB", "OUT_DB"]
    weights = [float(class_probs.get(c, 0.0)) for c in classes]
    weights = [w / sum(weights) for w in weights]

    def _alloc_counts(total):
        exp = [total * w for w in weights]
        base = [int(math.floor(v)) for v in exp]
        for i in sorted(range(len(base)), key=lambda x: (-(exp[x] - base[x]), x))[:total - sum(base)]: base[i] += 1
        return {classes[i]: base[i] for i in range(len(classes))}

    global_want = _alloc_counts(N_rel)
    rem = {c: global_want[c] for c in classes}
    stat_desired = {}

    # 精确平分总需求量到各个决策时刻
    for t_k in T_list:
        need = int(M_k[t_k])
        base = [min(int(math.floor(need * weights[i])), rem[classes[i]]) for i in range(len(classes))]
        slots = need - sum(base)
        frac = [need * weights[i] - math.floor(need * weights[i]) for i in range(len(classes))]
        for i in sorted(range(len(classes)), key=lambda x: (-frac[x], x)):
            if slots > 0 and rem[classes[i]] > base[i]: base[i] += 1; slots -= 1
        while slots > 0:
            placed = False
            for i in sorted(range(len(classes)), key=lambda x: (-(rem[classes[x]] - base[x]), x)):
                if rem[classes[i]] > base[i]: base[i] += 1; slots -= 1; placed = True; break
            if not placed: break
        stat_desired[t_k] = {classes[i]: base[i] for i in range(len(classes))}
        for i in range(len(classes)): rem[classes[i]] -= base[i]

    used, events, event_id = set(), [], 1
    stat_real = {t: {c: 0 for c in classes} for t in T_list}

    for t_k in T_list:
        need = int(M_k[t_k])
        if need <= 0: continue

        cand = [c for c in customers if
                c["NODE_ID"] not in used and float(tau_ref.get(c["NODE_ID"], 1e9)) >= float(t_k) + float(delta_look_h)]
        if len(cand) < need: cand = [c for c in customers if
                                     c["NODE_ID"] not in used and float(tau_ref.get(c["NODE_ID"], 1e9)) >= float(t_k)]
        if len(cand) < need: cand = [c for c in customers if c["NODE_ID"] not in used]

        cand.sort(key=lambda x: int(x["NODE_ID"]))
        rng.shuffle(cand)
        cand.sort(key=lambda x: (-float(tau_ref.get(x["NODE_ID"], 1e9)), int(x["NODE_ID"])))

        cand_info = []
        for c in cand:
            home = _assign_home_base(float(c["X"]), float(c["Y"]), base_points, r_db_units)
            cand_info.append((int(c["NODE_ID"]), float(c["X"]), float(c["Y"]), home,
                              any(p[2] != home for p in base_points) if home is not None else False))

        pool_in = [x for x in cand_info if x[3] is not None]
        pool_cross = [x for x in cand_info if x[3] is not None and x[4]]

        def pick(pool, k):
            take = []
            for item in pool:
                if item[0] not in used:
                    used.add(item[0]);
                    take.append(item)
                    if len(take) >= k: break
            return take

        want = stat_desired[t_k]
        take_in = pick(pool_in, want["IN_DB"])
        take_cross = pick(pool_cross, want["CROSS_DB"])
        take_out = pick(cand_info, need - len(take_in) - len(take_cross))

        for nid, ox, oy, home, _ in take_in:
            bx, by, _ = next(p for p in base_points if p[2] == home)
            nx, ny = _sample_uniform_in_disk(rng, bx, by, r_db_units)
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid, "NEW_X": nx, "NEW_Y": ny,
                           "EVENT_CLASS": "IN_DB", "DELTA_AVAIL_H": rng.uniform(delta_avail_min_h, delta_avail_max_h)})
            stat_real[t_k]["IN_DB"] += 1
            event_id += 1

        for nid, ox, oy, home, _ in take_cross:
            # 【终极修复】：移除护栏！允许跳跃到任意基站（包含历史基站）
            # 保证 CROSS_DB 的数量与设定比例 100% 吻合！
            others = [p for p in base_points if p[2] != home]
            bx, by, _ = rng.choice(others)
            nx, ny = _sample_uniform_in_disk(rng, bx, by, r_db_units)
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid, "NEW_X": nx, "NEW_Y": ny,
                           "EVENT_CLASS": "CROSS_DB",
                           "DELTA_AVAIL_H": rng.uniform(delta_avail_min_h, delta_avail_max_h)})
            stat_real[t_k]["CROSS_DB"] += 1
            event_id += 1

        for nid, ox, oy, home, _ in take_out:
            nx, ny = _sample_out_of_cover(rng, base_points, r_db_units, float(visual_range))
            events.append({"EVENT_ID": event_id, "EVENT_TIME": float(t_k), "NODE_ID": nid, "NEW_X": nx, "NEW_Y": ny,
                           "EVENT_CLASS": "OUT_DB", "DELTA_AVAIL_H": rng.uniform(delta_avail_min_h, delta_avail_max_h)})
            stat_real[t_k]["OUT_DB"] += 1
            event_id += 1

    events.sort(key=lambda e: (e["EVENT_TIME"], e["EVENT_ID"]))

    print("\n[EVENTS] 各决策点事件类别统计 (期望 vs 实际):")
    for t in T_list:
        d = stat_desired[t]
        r = stat_real[t]
        print(f"  t={t}h: 期望(IN={d['IN_DB']}, CROSS={d['CROSS_DB']}, OUT={d['OUT_DB']})  "
              f"-> 实际(IN={r['IN_DB']}, CROSS={r['CROSS_DB']}, OUT={r['OUT_DB']})")

    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EVENT_ID", "EVENT_TIME", "NODE_ID", "NEW_X", "NEW_Y", "EVENT_CLASS",
                                          "DELTA_AVAIL_H"])
        w.writeheader()
        for e in events:
            w.writerow({"EVENT_ID": e["EVENT_ID"], "EVENT_TIME": f"{e['EVENT_TIME']:.2f}", "NODE_ID": e["NODE_ID"],
                        "NEW_X": f"{e['NEW_X']:.3f}", "NEW_Y": f"{e['NEW_Y']:.3f}", "EVENT_CLASS": e["EVENT_CLASS"],
                        "DELTA_AVAIL_H": f"{e['DELTA_AVAIL_H']:.2f}"})


def generate_nodes_and_events(cfg: GenConfig, out_nodes_csv: str, out_events_csv: str, *, rho_rel: float,
                              decision_times: List[float], delta_look_h: float, class_probs: dict,
                              include_central_as_base: bool):
    generate_instance(cfg, out_nodes_csv)
    seed_for_events = int(cfg.seed) + 1000003
    generate_events_csv(out_nodes_csv, out_events_csv, rho_rel=rho_rel, decision_times=decision_times,
                        delta_look_h=delta_look_h, class_probs=class_probs,
                        include_central_as_base=include_central_as_base, seed_for_events=seed_for_events,
                        units_per_km=cfg.units_per_km, truck_speed_kmh=cfg.truck_speed_kmh,
                        truck_road_factor=cfg.truck_road_factor, drone_speed_kmh=cfg.drone_speed_kmh,
                        drones_per_base=cfg.drones_per_base, drone_roundtrip_km=cfg.drone_roundtrip_km,
                        visual_range=cfg.visual_range)

    meta_path = out_events_csv.replace(".csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.datetime.now().isoformat(), "cfg": asdict(cfg)}, f, indent=2)


# =======================================================
# 自动化执行入口 (带完美的目录分层 + 动态决策时间配置)
# =======================================================
if __name__ == "__main__":
    # 配置实验规模与种子
    scales = [25, 50, 100, 200]
    seed_list = [2021, 2022, 2023, 2024, 2025]

    # 核心扰动与事件参数
    rho_rel = 0.2
    delta_look_h = 0.25  # 恢复合理的提前量缓冲
    class_probs = {"IN_DB": 0.5, "CROSS_DB": 0.3, "OUT_DB": 0.2}  # 恢复正确的比例！
    include_central_as_base = True

    SCALE_CONFIGS = {
        25: {"v_range": 100.0, "db_cnt": 3, "decision_times": [1, 2]},
        50: {"v_range": 100.0, "db_cnt": 6, "decision_times": [1, 2, 3, 4]},
        100: {"v_range": 150.0, "db_cnt": 8, "decision_times": [1, 2, 3, 4, 5, 6]},
        200: {"v_range": 200.0, "db_cnt": 12, "decision_times": [1, 2, 3, 4, 5, 6, 7, 8]}
    }

    # 1. 建立顶级数据输出目录
    base_dir = "datasets_promise"
    os.makedirs(base_dir, exist_ok=True)

    print(f"🚀 开始批量生成结构化基准数据集...")

    for n_customers in scales:
        conf = SCALE_CONFIGS.get(n_customers)
        v_range = conf["v_range"]
        db_cnt = conf["db_cnt"]
        decision_times = conf["decision_times"]

        # 2. 为当前规模建立独立子文件夹
        scale_dir = os.path.join(base_dir, f"{n_customers}_data")
        os.makedirs(scale_dir, exist_ok=True)

        for s in seed_list:
            # 3. 为当前种子建立终端文件夹
            seed_dir = os.path.join(scale_dir, str(s))
            os.makedirs(seed_dir, exist_ok=True)

            cfg = GenConfig(
                n_customers=n_customers,
                visual_range=v_range,
                drone_roundtrip_km=10.0,
                truck_road_factor=1.5,
                seed=int(s),
                base_count_override={n_customers: db_cnt},
            )

            # 4. 文件命名抛弃时间戳
            out_nodes = os.path.join(seed_dir, f"nodes_{n_customers}_seed{s}.csv")
            out_events = os.path.join(seed_dir, f"events_{n_customers}_seed{s}.csv")

            print(f"\n[GENERATING] 规模: {n_customers} | 种子: {s} | 目录: {seed_dir}")

            generate_nodes_and_events(
                cfg, out_nodes, out_events,
                rho_rel=rho_rel,
                decision_times=decision_times,
                delta_look_h=delta_look_h,
                class_probs=class_probs,
                include_central_as_base=include_central_as_base,
            )

    print("\n🎉 结构化数据集生成完毕！请进入 datasets_promise/ 目录查收。")