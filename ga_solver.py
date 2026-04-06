# ga_solver.py
import random
import copy
import math
import simulation as sim


def _get_feasible_bases_strict(data, cid, bases_to_visit, visited_bases, base_onhand, drone_range):
    """
    [严谨版] 获取客户 cid 可用的无人机基站列表
    规则：
    1. 若基站 b 在 bases_to_visit (未来可达)：只要距离覆盖即可用。
    2. 若基站 b 在 visited_bases (历史已访)：必须 cid 在 base_onhand[b] (货在基站) 且距离覆盖才可用。
    """
    candidates = []

    # 1. 未来基站：默认可用 (只要覆盖)
    if bases_to_visit:
        candidates.extend(bases_to_visit)

    # 2. 历史基站：仅当货在手上时可用
    if visited_bases and base_onhand:
        for b in visited_bases:
            # base_onhand 是 dict: base_idx -> list/set of customer_idx
            onhand_customers = base_onhand.get(b, set())
            if cid in onhand_customers:
                candidates.append(b)

    # 3. 几何覆盖筛选
    feas = []
    x, y = float(data.nodes[cid]['x']), float(data.nodes[cid]['y'])
    # 去重 (防止 central 同时出现在 visited 和 to_visit)
    for b in sorted(list(set(candidates))):
        bx, by = float(data.nodes[b]['x']), float(data.nodes[b]['y'])
        d = math.hypot(x - bx, y - by)
        if 2.0 * d <= drone_range + 1e-9:
            feas.append(b)

    return feas


def _decode_and_repair(ind, start_idx, end_idx, data, visited_bases,
                       # 新增参数，用于 Smart Repair 评估
                       sim_module=None, ctx=None):
    """
    [Smart Repair 版解码器]
    在插入必须访问的基站时，不再只看距离最短，而是看 (Cost + Penalty) 最小。
    这能防止 GA 将基站插得太晚导致严重迟到。
    """
    perm, assign = ind
    route = [start_idx]
    b2d = {}

    # 1. 初始构建
    for cid in perm:
        if assign[cid] == 'truck':
            route.append(cid)
        else:
            base = assign[cid]
            b2d.setdefault(base, []).append(cid)

    if route[-1] != end_idx:
        route.append(end_idx)

    # 2. 识别缺失基站
    active_bases = set(b2d.keys())
    route_set = set(route)
    visited_set = set(visited_bases) if visited_bases else set()

    # 保证顺序确定性
    bases_to_insert = sorted([b for b in active_bases if (b not in route_set) and (b not in visited_set)])

    # 如果没有要插入的，直接返回
    if not bases_to_insert:
        return route, b2d

    # 准备评估环境 (如果未传入 sim 或 context，回退到 Dumb Distance Repair)
    use_smart_repair = (sim_module is not None) and (ctx is not None)

    if use_smart_repair:
        truck_speed = sim_module.TRUCK_SPEED_UNITS
        drone_speed = sim_module.DRONE_SPEED_UNITS
        alpha = 0.3
        lam = ctx.get('lambda_late', 50.0)
        arr_pref = ctx.get('arrival_prefix', None)

    # 3. 逐个插入缺失基站
    for b_idx in bases_to_insert:
        best_pos = -1
        best_obj = float('inf')

        # 遍历所有可行插入位置 (1 到 len-1)
        # 注意：这里计算量是 O(N_bases * N_nodes)，对于 GA 种群可能略慢，但对于准确性至关重要
        for i in range(1, len(route)):
            # 构造临时路线
            cand_route = route[:i] + [b_idx] + route[i:]

            obj_val = float('inf')

            if use_smart_repair:
                # [Smart] 调用 sim 评估完整 Cost + Penalty
                # 此时 b2d 已经包含了该基站的任务
                res = sim_module.evaluate_truck_drone_with_time(
                    data, cand_route, b2d,
                    start_time=ctx.get('start_time', 0.0),
                    truck_speed=truck_speed, drone_speed=drone_speed,
                    alpha_drone=alpha, lambda_late=lam, arrival_prefix=arr_pref
                )
                # res[0] 是 cost (含 penalty 吗？取决于 sim 实现，通常 sim 返回的是 components)
                # 假设 sim 返回 (cost, t_d, d_d, t_l, d_l, total_l, ...)
                # 且 sim 内部 cost = base + lam * late
                obj_val = res[0]
            else:
                # [Dumb] 回退到纯距离增量 (防止 crash)
                prev, curr = route[i - 1], route[i]
                d1 = sim_module.truck_arc_cost(data, prev, b_idx)
                d2 = sim_module.truck_arc_cost(data, b_idx, curr)
                d3 = sim_module.truck_arc_cost(data, prev, curr)
                obj_val = d1 + d2 - d3

            if obj_val < best_obj:
                best_obj = obj_val
                best_pos = i

        # 执行最佳插入
        if best_pos != -1:
            route.insert(best_pos, b_idx)

    return route, b2d

def ga_truck_drone(data, base_to_drone_customers, max_iter=200, pop_size=50,
                   mutation_rate=0.2, crossover_rate=0.8, alpha_drone=0.3,
                   lambda_late=50.0, truck_customers=None, start_idx=None,
                   start_time=0.0, bases_to_visit=None, ctx=None):
    """
    卡车-无人机协同遗传算法 (GA) - Strict Alignment Version
    """
    if ctx is None: ctx = {}

    # [Fix Risk 5] 注入种子，保证可复现性
    seed = ctx.get('seed', None)
    if seed is not None:
        random.seed(seed)

    # 上下文参数提取
    drone_range = ctx.get("drone_range", sim.DRONE_RANGE_UNITS)
    truck_speed = sim.TRUCK_SPEED_UNITS
    drone_speed = sim.DRONE_SPEED_UNITS
    arrival_prefix = ctx.get("arrival_prefix", None)

    # [Fix Risk 3] 读取硬迟到阈值 (必须与 ALNS 对齐)
    # 默认值 0.5 小时，必须与 main.py 配置一致
    late_hard = float(ctx.get("late_hard", 0.5))
    max_no_improve = int(ctx.get("max_no_improve", 50))
    # [Fix Risk 1] 获取 visited 和 onhand 信息
    visited_bases = ctx.get("visited_bases", [])
    visited_bases_set = set(visited_bases)
    base_onhand = ctx.get("base_onhand", {})  # dict: base -> set(cids)

    if start_idx is None:
        start_idx = data.central_idx
    end_idx = data.central_idx

    # 1) bases_to_visit 强校验与 fallback (Fix User Request #1)
    if bases_to_visit is None and ctx is not None:
        bases_to_visit = ctx.get("bases_to_visit")

    if bases_to_visit is None:
        # 方案 A: 强制报错，防止默默退化为仅 central
        raise ValueError(
            "[GA-FATAL] bases_to_visit is None! GA will degrade to TruckOnly. Please pass correct bases list.")

    # Debug 打印
    if ctx.get("verbose", False):
        print(f"[GA-DBG] bases_to_visit count={len(bases_to_visit)}, first5={bases_to_visit[:5]}")

    if truck_customers is None: truck_customers = []

    # 2) allowed_customers 严格过滤 (Fix User Request #2)
    # 必须排除 base/central 节点，否则 GA 会将其视为客户进行排列，导致解码异常
    _drone_custs = {c for cs in base_to_drone_customers.values() for c in cs}
    raw_allowed = set(truck_customers) | _drone_custs
    allowed_customers = []
    _ignored_nodes = []

    for c in raw_allowed:
        # 严格检查 node_type
        ntype = str(data.nodes[c].get("node_type", "")).lower()
        if ntype == "customer":
            allowed_customers.append(c)
        else:
            _ignored_nodes.append(c)

    allowed_customers.sort()

    if _ignored_nodes:
        print(
            f"[WARN-GA] Filtered out non-customer nodes from allowed set: {_ignored_nodes} (Types: {[data.nodes[i].get('node_type') for i in _ignored_nodes]})")

    if not allowed_customers:
        return [start_idx, end_idx], {}, 0.0, 0.0, 0.0, 0.0, 0.0
    # 预处理：识别 force_truck 和 可行基站
    cust_info = {}
    for cid in allowed_customers:
        node = data.nodes[cid]
        # [Fix Risk 1] 使用严格的可行基站筛选逻辑
        feas_bases = _get_feasible_bases_strict(
            data, cid, bases_to_visit, visited_bases, base_onhand, drone_range
        )

        locked = node.get("base_lock", None)
        if locked is not None:
            # 如果锁定的基站不可行 (e.g. 已经路过且没货)，则只能 force truck
            if locked in feas_bases:
                feas_bases = [locked]
            else:
                feas_bases = []  # 强转卡车

        cust_info[cid] = {
            'force_truck': node.get("force_truck", 0) == 1,
            'feas_bases': feas_bases
        }
    # 3) 补齐 arrival_prefix (Fix User Request #3)
    # 确保 visited_bases 的发飞时间不缺失，否则评估函数会由默认值导致异常
    if arrival_prefix is None:
        arrival_prefix = {}
    else:
        arrival_prefix = dict(arrival_prefix)  # 浅拷贝避免副作用

    _patched_bases = []
    for b in visited_bases_set:
        if b not in arrival_prefix:
            arrival_prefix[b] = float(start_time)
            _patched_bases.append(b)

    if _patched_bases and ctx.get("verbose", False):
        print(f"[GA-DBG] Patched arrival_prefix for visited bases (set to t={start_time:.2f}): {_patched_bases}")

    # 4) late_hard 检查 (Fix User Request #4)
    if late_hard > 240.0:  # 经验值：如果大于 240h 可能配置有误
        print(f"[WARN-GA] late_hard={late_hard} seems very large. Check config.")
    # [核心修改] 评估函数：加入硬约束惩罚
    def evaluate(ind):
        # 1. 解码 (含修复逻辑)
        route, b2d = _decode_and_repair(ind, start_idx, end_idx, data, visited_bases_set, sim_module=sim, ctx={
                'start_time': start_time,
                'lambda_late': lambda_late,
                'arrival_prefix': arrival_prefix
            })

        # 2. 物理评估
        # res: (cost, truck_d, drone_d, t_late, d_late, total_late, t_time)
        res = sim.evaluate_truck_drone_with_time(
            data, route, b2d,
            start_time=start_time, truck_speed=truck_speed, drone_speed=drone_speed,
            alpha_drone=alpha_drone, lambda_late=lambda_late, arrival_prefix=arrival_prefix
        )

        # 3. [Fix Risk 3] 硬护栏检查 (Hard Constraint Check)
        # 如果总迟到超过 hard 阈值，强制将 cost 设为极大值 (Infeasible)
        # 这样 GA 的选择算子 (Selection) 会自然淘汰这些解
        raw_cost, t_d, d_d, t_l, d_l, total_l, t_time = res

        if total_l > late_hard:
            # 惩罚值 = 原始 Cost + 1e6 (确保比任何可行解都差)
            penalized_cost = raw_cost + 1.0e6
            # 返回修改后的元组 (cost 变大，其他指标保持以便 debug)
            return (penalized_cost, t_d, d_d, t_l, d_l, total_l, t_time)

        return res

    # =================================================
    # [Fix Risk 6] 热启动：注入 Greedy/Preplan 解 (防止 GA 跑输基线)
    # =================================================
    warm_start_ind = None
    init_route = ctx.get("init_route", None)

    if init_route:
        # 1. 提取 Greedy 解中的卡车客户顺序 (基因 Perm 的一部分)
        # 注意：init_route 包含 base/central，需提取 allowed_customers 里的 subset
        greedy_truck_seq = []
        _seen = set()
        for nid in init_route:
            if (nid in cust_info) and (nid not in _seen):
                greedy_truck_seq.append(nid)
                _seen.add(nid)

        # 2. 提取无人机客户 (拼接到 Perm 尾部)
        # (在 GA 编码中，无人机客户在 perm 中的位置不影响分配，只要在 perm 里就行)
        greedy_drone_seq = [c for c in allowed_customers if c not in _seen]

        # 3. 构造精英染色体 Perm
        elite_perm = greedy_truck_seq + greedy_drone_seq

        # 4. 构造精英分配 Assign
        # 默认全部 Truck，然后根据 base_to_drone_customers 修正为 Drone
        elite_assign = {c: 'truck' for c in allowed_customers}

        # 反查表：client -> base
        _c_to_b = {}
        for b, cs in base_to_drone_customers.items():
            for c in cs:
                _c_to_b[c] = b

        for c in allowed_customers:
            # 如果原解是无人机，且该基站当前仍可行，则保持无人机
            if c in _c_to_b:
                target_b = _c_to_b[c]
                if target_b in cust_info[c]['feas_bases']:
                    elite_assign[c] = target_b
            # 否则保持默认 'truck' (elite_assign初始化值)

        # 校验长度防崩
        if len(elite_perm) == len(allowed_customers):
            warm_start_ind = (elite_perm, elite_assign)
            if ctx.get("verbose", False):
                print(f"[GA-DBG] Injected Warm-Start solution (truck_len={len(greedy_truck_seq)})")
    # 3. 初始化种群
    population = []
    # 如果有热启动解，先加进去
    if warm_start_ind:
        population.append(copy.deepcopy(warm_start_ind))

    # 补齐剩余个体 (随机)
    while len(population) < pop_size:
        # 3.1 随机排列
        perm = allowed_customers[:]
        random.shuffle(perm)

        # 3.2 随机分配
        assign = {}
        for cid in allowed_customers:
            info = cust_info[cid]
            if info['force_truck'] or not info['feas_bases']:
                assign[cid] = 'truck'
            else:
                # 随机决定：卡车 vs 可用基站
                choices = ['truck'] + info['feas_bases']
                assign[cid] = random.choice(choices)
        population.append((perm, assign))

    best_ind = population[0]
    best_eval = evaluate(best_ind)
    no_improve_cnt = 0
    # 遗传算子
    def crossover_ox(p1_perm, p2_perm):
        size = len(p1_perm)
        if size < 2: return p1_perm[:]
        a, b = sorted(random.sample(range(size), 2))
        child_perm = [None] * size
        child_perm[a:b] = p1_perm[a:b]
        pointer = b
        for item in p2_perm:
            if item not in child_perm:
                if pointer >= size: pointer = 0
                child_perm[pointer] = item
                pointer += 1
        return child_perm

    def mutate(ind):
        perm, assign = list(ind[0]), dict(ind[1])
        # 排列变异
        if len(perm) >= 2 and random.random() < mutation_rate:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]

        # 分配变异
        if random.random() < mutation_rate:
            cid = random.choice(allowed_customers)
            info = cust_info[cid]
            # 只有非强制卡车且有可行基站时才能变异
            if not info['force_truck'] and info['feas_bases']:
                choices = ['truck'] + info['feas_bases']
                # 尝试变异成不同的模式
                if assign[cid] in choices and len(choices) > 1:
                    choices.remove(assign[cid])
                assign[cid] = random.choice(choices)
        return (perm, assign)

    # 4. 进化主循环
    for gen in range(max_iter):
        pop_evals = [(ind, evaluate(ind)) for ind in population]
        # 按 cost 排序 (res[0])
        pop_evals.sort(key=lambda x: x[1][0])

        # [修改] 检查是否有改进 (使用 epsilon 避免浮点误差)
        # 如果当前代的最优解比历史最优解更好
        if pop_evals[0][1][0] < best_eval[0] - 1e-6:
            best_ind = pop_evals[0][0]
            best_eval = pop_evals[0][1]
            no_improve_cnt = 0  # 重置计数器
        else:
            no_improve_cnt += 1  # 计数器累加

        # [新增] 早停检查
        if no_improve_cnt >= max_no_improve:
            if ctx.get("verbose", False):
                print(f"[GA-STOP] Early stopping at gen {gen} (no improve for {max_no_improve} gens)")
            break

        next_gen = [pop_evals[0][0]]  # 精英保留

        # 锦标赛选择
        def select():
            candidates = random.sample(pop_evals, min(3, len(pop_evals)))
            return min(candidates, key=lambda x: x[1][0])[0]

        while len(next_gen) < pop_size:
            p1, p2 = select(), select()
            if random.random() < crossover_rate:
                c1_perm = crossover_ox(p1[0], p2[0])
                # 均匀交叉分配向量
                c1_assign = {c: p1[1][c] if random.random() < 0.5 else p2[1][c] for c in allowed_customers}
                # 校验交叉后的分配是否合法 (防止基站不可行) - 实际上 cust_info 约束是静态的，只要 p1 p2 合法，子代通常合法
                child = (c1_perm, c1_assign)
            else:
                child = copy.deepcopy(p1)
            child = mutate(child)
            next_gen.append(child)

        population = next_gen[:pop_size]

    # 5. 返回结果 (Robust Return)
    # [Fix Risk 4] 显式解包，防止 sim 接口变动导致下标越界
    final_route, final_b2d = _decode_and_repair(best_ind, start_idx, end_idx, data, visited_bases_set, sim_module=sim, ctx={
            'start_time': start_time,
            'lambda_late': lambda_late,
            'arrival_prefix': arrival_prefix
        })
    cost, truck_d, drone_d, truck_l, drone_l, total_l, total_time = best_eval

    return (final_route, final_b2d, cost, truck_d, drone_d, total_l, total_time)