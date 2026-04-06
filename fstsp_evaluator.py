# -*- coding: utf-8 -*-
import math


def euclid(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def evaluate_fstsp_system(
        data,
        full_route_ids,
        fstsp_triplets,
        truck_speed_units=150.0,
        drone_speed_units=300.0,
        truck_road_factor=1.5,
        alpha_drone=0.3,
        lambda_late=50.0):
    """
    顶刊级 FSTSP 系统评估器
    严格处理卡车在回收点的等待（Time Synchronization），计算绝对准确的延迟和系统完成时间。
    """
    if not full_route_ids:
        return {}

    # 1. 预处理三元组，建立【回收点 -> (起飞点, 客户点)】的映射
    recoveries = {}
    for launch_id, cust_id, rend_id in fstsp_triplets:
        recoveries.setdefault(rend_id, []).append((launch_id, cust_id))

    arrival_times = {}
    first_depart_times = {}  # 记录节点首次离开的时间，用于无人机起飞
    finish_times = {}

    truck_dist = 0.0
    drone_dist = 0.0

    depart_t_prev = 0.0

    # 2. 严格按顺序模拟时空演进
    for idx, curr in enumerate(full_route_ids):
        # A. 卡车到达时间
        if idx == 0:
            arr_t = 0.0
        else:
            prev = full_route_ids[idx - 1]
            node_prev = data.nodes[prev]
            node_curr = data.nodes[curr]
            dist_unit = euclid(node_prev['x'], node_prev['y'], node_curr['x'], node_curr['y'])
            arc_dist = dist_unit * truck_road_factor

            truck_dist += arc_dist
            travel_time = arc_dist / truck_speed_units
            arr_t = depart_t_prev + travel_time

        # 记录最新到达时间（如果是中心仓库被多次访问，保留最后返回的时间）
        arrival_times[curr] = arr_t
        dep_t = arr_t

        # B. 核心：如果该点有无人机回收，卡车必须等待！
        if curr in recoveries:
            for launch_id, cust_id in recoveries[curr]:
                # 无人机的起飞时间，是卡车【首次】离开 launch_id 的时间
                l_time = first_depart_times.get(launch_id, 0.0)

                node_l = data.nodes[launch_id]
                node_c = data.nodes[cust_id]
                node_r = data.nodes[curr]

                d_out = euclid(node_l['x'], node_l['y'], node_c['x'], node_c['y'])
                d_in = euclid(node_c['x'], node_c['y'], node_r['x'], node_r['y'])

                drone_dist += (d_out + d_in)

                # 无人机服务客户的时间
                service_t = l_time + (d_out / drone_speed_units)
                # 无人机抵达回收点的时间
                drone_arr_t = service_t + (d_in / drone_speed_units)

                finish_times[cust_id] = service_t

                # 【时空同步】：卡车离开时间必须晚于无人机归来时间！
                dep_t = max(dep_t, drone_arr_t)

        # 记录首次离开时间
        if curr not in first_depart_times:
            first_depart_times[curr] = dep_t

        depart_t_prev = dep_t

    # 3. 计算准确的迟到惩罚
    truck_late = 0.0
    drone_late = 0.0

    for idx in full_route_ids:
        node = data.nodes[idx]
        if str(node.get('node_type', '')).lower() == 'customer':
            due = float(node.get('effective_due', node.get('due_time', 0.0)))
            arr = arrival_times[idx]
            finish_times[idx] = arr
            if arr > due:
                truck_late += (arr - due)

    for launch_id, cust_id, rend_id in fstsp_triplets:
        node_c = data.nodes[cust_id]
        due = float(node_c.get('effective_due', node_c.get('due_time', 0.0)))
        fin = finish_times[cust_id]
        if fin > due:
            drone_late += (fin - due)

    total_late = truck_late + drone_late
    cost = truck_dist + alpha_drone * drone_dist + lambda_late * total_late
    sys_time = depart_t_prev  # 系统的最终完成时间，是卡车最后离开终点（并回收完所有无人机）的时间

    return {
        "cost": cost,
        "cost(obj)": cost,
        "truck_dist": truck_dist,
        "truck_dist_eff": truck_dist,
        "drone_dist": drone_dist,
        "truck_late": truck_late,
        "drone_late": drone_late,
        "total_late": total_late,
        "system_time": sys_time,
        "truck_total_time": sys_time,
        "arrival_times": arrival_times,
        "finish_times": finish_times,
        "arrival": arrival_times,
        "depart": {},
        "finish": finish_times
    }