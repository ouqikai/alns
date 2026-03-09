# vns_solver.py
import random
import copy
import simulation as sim
import operators as ops


def vns_truck_drone(data, base_to_drone_customers, max_iter=200, alpha_drone=0.3,
                    lambda_late=50.0, truck_customers=None, start_idx=None,
                    start_time=0.0, bases_to_visit=None, ctx=None):
    """
    变邻域搜索 (Variable Neighborhood Search, VNS) 基线算法
    为了保证公平对比，使用与 ALNS 相同的算子，但按 VNS 的逻辑组织邻域。
    """
    if ctx is None: ctx = {}

    # 提取参数
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)
    truck_speed = sim.TRUCK_SPEED_UNITS
    drone_speed = sim.DRONE_SPEED_UNITS
    arrival_prefix = ctx.get("arrival_prefix", None)

    if start_idx is None:
        start_idx = data.central_idx
    if bases_to_visit is None:
        bases_to_visit = [i for i, n in enumerate(data.nodes) if n.get("node_type") == "base"]
        if data.central_idx not in bases_to_visit:
            bases_to_visit.append(data.central_idx)

    if truck_customers is None: truck_customers = []

    # 1. 构造初始解 (与 ALNS 同源)
    init_route = ctx.get("init_route", None)
    if init_route is not None and len(init_route) >= 2:
        current_route = list(init_route)
    else:
        current_route = ops.nearest_neighbor_route_truck_only(
            data, truck_customers, start_idx=start_idx, end_idx=data.central_idx,
            bases_to_visit=bases_to_visit
        )
    current_b2d = {b: lst[:] for b, lst in base_to_drone_customers.items()}

    def _evaluate(r, b2d_dict):
        return sim.evaluate_truck_drone_with_time(
            data, r, b2d_dict, start_time=start_time, truck_speed=truck_speed,
            drone_speed=drone_speed, alpha_drone=alpha_drone, lambda_late=lambda_late,
            arrival_prefix=arrival_prefix
        )

    current_eval = _evaluate(current_route, current_b2d)
    current_cost = current_eval[0]

    best_route = current_route[:]
    best_b2d = {b: lst[:] for b, lst in current_b2d.items()}
    best_eval = current_eval

    # 2. 定义 VNS 邻域结构 (Neighborhoods)
    # 我们将 ALNS 的 Destroy-Repair 组合包装成 VNS 的邻域
    # 邻域从小到大，从温和到剧烈
    neighborhoods = [
        # N1: 移除 2 个点并贪婪修复
        {"remove": 2, "D": ops.D_worst_route, "R": ops.R_greedy_then_drone},
        # N2: 移除 4 个点并后悔修复
        {"remove": 4, "D": ops.D_random_route, "R": ops.R_regret_then_drone},
        # N3: 针对位置变更的聚焦扰动
        {"remove": 3, "D": ops.D_reloc_focus_v2, "R": ops.R_base_feasible_drone_first},
        # N4: 强力覆盖切换
        {"remove": 5, "D": ops.D_switch_coverage, "R": ops.R_regret_then_drone},
        # N5: 迟到专项大尺度扰动
        {"remove": 6, "D": ops.D_late_worst, "R": ops.R_late_repair_reinsert}
    ]
    K_max = len(neighborhoods)

    k = 0
    no_improve_cnt = 0
    max_no_improve = ctx.get("max_no_improve", 200)

    # 3. VNS 主循环
    for iteration in range(max_iter):
        # Shaking & Local Search (在 ALNS 算子体系中，Destroy=Shaking, Repair=Local Search)
        nh = neighborhoods[k]

        # 准备上下文
        vns_ctx = dict(ctx)
        vns_ctx["num_remove"] = nh["remove"]
        vns_ctx["protected_nodes"] = set(bases_to_visit) | {start_idx, data.central_idx}

        # 扰动 (Destroy)
        cand_route, cand_b2d, removed = nh["D"](data, current_route, current_b2d, vns_ctx)
        # 修复 (Repair)
        cand_route, cand_b2d = nh["R"](data, cand_route, cand_b2d, removed, vns_ctx)

        # 兜底约束
        if cand_route is None: continue
        cand_route, cand_b2d = sim.enforce_force_truck_solution(data, cand_route, cand_b2d)

        # 评估
        cand_eval = _evaluate(cand_route, cand_b2d)
        cand_cost = cand_eval[0]

        # VNS 接受准则：只接受更好的解 (下降法)
        # 注意：标准 VNS 不使用模拟退火的概率接受
        if cand_cost < current_cost - 1e-6:
            # 改进了：更新当前解，并回到第一个邻域
            current_route = cand_route
            current_b2d = cand_b2d
            current_cost = cand_cost
            current_eval = cand_eval
            k = 0  # 回到最小邻域

            if cand_cost < best_eval[0] - 1e-6:
                best_route = current_route[:]
                best_b2d = {b: lst[:] for b, lst in current_b2d.items()}
                best_eval = current_eval
                no_improve_cnt = 0
        else:
            # 没改进：进入下一个更大的邻域
            k = (k + 1) % K_max
            no_improve_cnt += 1

        if no_improve_cnt >= max_no_improve:
            break

    # 最终约束兜底
    best_route, best_b2d = sim.enforce_force_truck_solution(data, best_route, best_b2d)
    cost, truck_d, drone_d, _, _, total_l, total_time = best_eval

    return best_route, best_b2d, cost, truck_d, drone_d, total_l, total_time