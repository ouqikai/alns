"""
单文件求解器（清理版）：单卡车 + 多无人机 + 客户位置在线变更（不含回放模块）

设计目标：
- 保持原有算法/输出口径不变（关键指标与旧版输出口径兼容）
- 在不引入新依赖的前提下，收敛重复逻辑并清理历史遗留实现

文件结构（分区在代码中用大标题标注）：
1) 全局常量与随机种子
2) 在线扰动日志（保存/回放）
3) 基础工具（距离/时间窗/调试打印）
4) 评估与调度（卡车时刻表、多无人机调度、成本合成）
5) 在线扰动（申请生成/回放/应用）
6) ALNS 算子与主循环
7) 可视化与统计输出
8) 实验入口（单次/套件/静态对比/CLI）

说明：本文件仍保持可直接运行（main()/main_cli()），便于你现有实验脚本与复现流程不改。
"""
# alns_truck__drone（new）.py
import math
import random
import os
import operators as ops  # 引入算子模块
DEBUG_QUICK_FILTER = True
import simulation as sim  # 引入仿真模块
import utils as ut # 工具模块
import dynamic_logic as dyn  # 动态逻辑
# 数据读入：强制依赖 data_io_1322.py（不再提供单文件兜底，避免掩盖 import 错误）
from data_io_1322 import read_data, Data, recompute_cost_and_nearest_base
# 引入拆分出去的可视化模块
from viz_utils import visualize_truck_drone, compute_global_xlim_ylim, _normalize_decisions_for_viz

# ========= 数据集 Schema（节点 + 动态事件）=========
# 说明：nodes.csv 仅包含静态节点字段；动态请求流由 events.csv 单独给出（EVENT_TIME, NODE_ID, NEW_X, NEW_Y, EVENT_CLASS）。
CSV_REQUIRED_COLS = [
    "NODE_ID","NODE_TYPE","ORIG_X","ORIG_Y","DEMAND","READY_TIME","DUE_TIME"
]
CSV_NODE_TYPES = {"central", "base", "customer"}
EPS = 1e-9
# 四种算子（动作）
DEBUG = False
# 中文注释：迟到定位开关（只建议临时打开，避免刷屏）
DEBUG_LATE = True
DEBUG_LATE_TOPK = 15
DEBUG_LATE_SCENES = None   # 只看 scene=0；想看全部就改成 None
# 中文注释：重插入诊断开关（仅用于定位“某个迟到客户为什么迟到”）
DEBUG_REINS_DIAG = True
# 中文注释：指定某个客户 idx；None 表示自动选择“当前最迟到的卡车客户”
DEBUG_REINS_CID = None

CFG_A = {
    "NAME": "A_paired_baseline",
    "PAIRING_MODE": "paired",
    "late_hard": 0.12,  # 建议显式写在 cfg 里（要更严就 0.10）
    # 固定两对：等价于你原来“捆绑”的两套组合
    "ALLOWED_PAIRS": [
        ("D_random_route", "R_greedy_then_drone"),
        ("D_worst_route",  "R_regret_then_drone"),
    ],
    "DESTROYS": ["D_random_route", "D_worst_route"],
    "REPAIRS":  ["R_greedy_then_drone", "R_regret_then_drone"],
}

CFG_D = {
    "NAME": "D_full_structured",
    "PAIRING_MODE": "free",
    "late_hard": 0.12,  # 建议显式写在 cfg 里（要更严就 0.10）
    "DESTROYS": ["D_random_route", "D_worst_route", "D_reloc_focus_v2", "D_switch_coverage"],
    "REPAIRS":  ["R_greedy_only", "R_regret_only", "R_greedy_then_drone", "R_regret_then_drone", "R_late_repair_reinsert", "R_base_feasible_drone_first"],
}

def dprint(*args, **kwargs):
    """统一的调试打印开关，避免到处散落 print"""
    if DEBUG:
        print(*args, **kwargs)

# ==============
# 一些基础函数
# ==============
def alns_truck_drone(data, base_to_drone_customers, max_iter=200, remove_fraction=0.1, T_start=1.0,
                     T_end=0.01, alpha_drone=0.3, lambda_late=50.0, truck_customers=None, use_rl=False,
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

    if ctx.get("verbose", False):
        print(f"初始: 成本={current_cost:.3f}, 卡车距={current_truck_dist:.3f}, 无人机距={current_drone_dist:.3f}, 总迟到={current_total_late:.3f}")

    # === RL: 每个算子的得分（Q 值） ===

    n_inner = len(current_route) - 2
    if n_inner <= 0:
        # 兜底：仍需保证 force_truck 约束一致
        best_route, best_b2d = sim.enforce_force_truck_solution(data, best_route, best_b2d)
        return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time
    # ---------- ctx 处理：保留调用方传入的 dict 引用，用于回传算子统计 ----------
    ctx_in = ctx if isinstance(ctx, dict) else {}
    ctx = ctx_in
    ctx = build_ab_cfg(ctx)

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

    # if bool(ctx.get("verbose_stat", True)):
    #     _print_operator_stats(op_stat, top_k=int(ctx.get("verbose_stat_topk", 30)))
    return best_route, best_b2d, best_cost, best_truck_dist, best_drone_dist, best_total_late, best_truck_time


def run_one(file_path: str, seed: int, ab_cfg: dict, perturbation_times=None, enable_plot: bool = False, verbose: bool = True, events_path: str = "", decision_log_path: str = ""):
    if verbose:
        print("[CFG-IN]", ab_cfg.get("PAIRING_MODE"), ab_cfg.get("late_hard"),
          len(ab_cfg.get("DESTROYS", [])), len(ab_cfg.get("REPAIRS", [])))

    if perturbation_times is None:
        perturbation_times = []
    # 统一过滤/去重/排序，避免传入 0 导致重复场景、以及不同运行方式输出不一致
    perturbation_times = _normalize_perturbation_times(perturbation_times)
    def cover_uncovered_by_truck_suffix(data, full_route, full_b2d, prefix_len, verbose=False):
        """
        动态拼接后工程兜底：若出现未覆盖客户（既不在卡车路径也不在无人机任务），
        则把它们强制插入到“后缀部分”的卡车路径中（不动前缀已执行部分），避免丢点崩溃。
        """
        all_customers = {i for i, n in enumerate(data.nodes) if n.get("node_type") == "customer"}
        truck_customers = {i for i in full_route
                           if 0 <= i < len(data.nodes) and data.nodes[i].get("node_type") == "customer"}
        drone_customers = {c for cs in full_b2d.values() for c in cs}
        uncovered = sorted(all_customers - (truck_customers | drone_customers))
        if not uncovered:
            return full_route

        if verbose:
            nids = [data.nodes[i].get("node_id", i) for i in uncovered]
            print(f"[GUARD][COVER] 发现未覆盖客户 {len(uncovered)} 个，强制插入卡车后缀路径 node_id={nids}")

        r = full_route[:]
        if len(r) < 2:
            r.extend(uncovered)
            return r

        start_pos = max(int(prefix_len), 1)
        end_pos = max(len(r) - 1, start_pos)

        def delta_cost(route, node_idx, pos):
            a = route[pos - 1]
            b = route[pos]
            return float(sim.truck_arc_cost(data, a, node_idx) + sim.truck_arc_cost(data, node_idx, b) - sim.truck_arc_cost(data, a, b))

        for c in uncovered:
            # 标记 force_truck，避免后续又被分给无人机
            try:
                data.nodes[c]["force_truck"] = 1
            except Exception:
                pass

            best_pos = end_pos
            best_delta = float("inf")
            for pos in range(start_pos, end_pos + 1):
                try:
                    d = delta_cost(r, c, pos)
                except Exception:
                    continue
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            r.insert(best_pos, c)

        return r

    if seed is not None:
        ut.set_seed(int(seed))

    ab_cfg = build_ab_cfg(ab_cfg)
    # ===================== 1) 读取数据（场景0：全原始坐标）=====================
    data = read_data(file_path, scenario=0, strict_schema=True)
    if verbose:
        ut.print_tw_stats(data)  # 或者 print_tw_stats(data_cur)
    # 可选：schema 对齐检查
    try:
        if hasattr(data, "schema_cols") and "CSV_REQUIRED_COLS" in globals():
            # 中文注释：允许 nodes.csv 额外列存在（例如 *_promise.csv 的 PROM_* 列），只要必需列不缺失即可。
            _cols = list(getattr(data, "schema_cols", []) or [])
            _missing = [c for c in CSV_REQUIRED_COLS if c not in _cols]
            if _missing:
                raise RuntimeError(f"数据 schema 缺失必需列: {_missing}；请检查 data_io 的 CSV_REQUIRED_COLS")
    except Exception as e:
        raise

    if verbose:
        print(f"节点数: {len(data.nodes)}, 中心仓库 idx: {data.central_idx}")

    # ===================== [OFFLINE-EVENTS] 读取 events.csv（若提供）=====================
    nodeid2idx = {int(n.get("node_id")): i for i, n in enumerate(data.nodes)}
    offline_events = []
    offline_groups = None
    decision_log_rows = []
    if events_path:
        try:
            offline_events = ut.load_events_csv(events_path)
        except Exception as _e:
            raise RuntimeError(f"[OFFLINE] events.csv 读取失败：{events_path}，err={_e}")
        if not offline_events:
            raise RuntimeError(f"[OFFLINE] events_path 提供但读取为空：{events_path}")
        offline_groups = ut.group_events_by_time(offline_events)
        if verbose:
            print(f"[OFFLINE] load events: {events_path}, events={len(offline_events)}")
        # 用 events.csv 中出现过的 EVENT_TIME 覆盖决策点集合（支持非连续/非整数）
        _ts = sorted({round(float(e.get('EVENT_TIME', 0.0)), 6) for e in offline_events})
        perturbation_times = [float(t) for t in _ts if float(t) > 0.0]
        if verbose:
            print(f"[OFFLINE] decision times overridden by events: T=1..{len(perturbation_times)}")
    # ===================== 2) 初始分类（场景0）=====================
    base_to_drone_customers, truck_customers = sim.classify_clients_for_drone(data)
    if verbose:
        print("需要卡车服务的客户数:", len(truck_customers))
        print("各基站无人机客户数:", {b: len(cs) for b, cs in base_to_drone_customers.items()})

    # ===================== 3) 场景0：跑一次 ALNS（No-RL）=====================
    if verbose:
        print("\n===== Advanced ALNS (No RL, official solution) =====")

    ctx0 = dict(ab_cfg)  # 关键：场景0也吃实验配置（paired/free/算子池）
    ctx0["verbose"] = verbose

    # [PROMISE] 场景0不计迟到：避免 late_hard 护栏误伤（即使你实验配置里开启了 late_hard）
    ctx0["late_hard"] = 1e18
    ctx0["late_hard_delta"] = 1e18

    (best_route,
     best_b2d,
     best_cost,
     best_truck_dist,
     best_drone_dist,
     best_total_late,
     best_truck_time) = alns_truck_drone(
        data,
        base_to_drone_customers,
        max_iter=1000,
        remove_fraction=0.1,
        T_start=1.0,
        T_end=0.01,
        alpha_drone=0.3,
        lambda_late=0.0,
        truck_customers=truck_customers,
        use_rl=False,
        rl_tau=0.5,
        rl_eta=0.1,
        ctx=ctx0
    )

    arrival_times, total_time, total_late = sim.compute_truck_schedule(
        data, best_route, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
    )
    depart_times, finish_times, base_finish_times = sim.compute_multi_drone_schedule(
        data, best_b2d, arrival_times,
        num_drones_per_base=sim.NUM_DRONES_PER_BASE,
        drone_speed=sim.DRONE_SPEED_UNITS
    )

    # ===================== [PROMISE] 3.5) 用场景0 ETA0 生成并冻结平台承诺窗 =====================
    # 中文注释：场景0不考虑时间窗/迟到，仅用于生成“平台承诺窗口”（PROM_READY/PROM_DUE），并冻结用于后续所有场景。
    # 护栏：若输入本身已经是 *_promise.csv，则认为承诺窗已冻结，避免再次生成并输出 _promise_promise.csv。
    if ut._is_promise_nodes_file(file_path):
        ut.freeze_existing_promise_windows_inplace(data)
        if verbose:
            print(f"[PROMISE] input already *_promise.csv, skip regenerate/write: {file_path}")
    else:
        _full_eval0_tmp = sim.evaluate_full_system(
            data, best_route, best_b2d,
            alpha_drone=0.3, lambda_late=0.0,
            truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
        )
        eta0_map = ut.compute_eta_map(data, best_route, best_b2d, _full_eval0_tmp, drone_speed=sim.DRONE_SPEED_UNITS)
        ut.apply_promise_windows_inplace(data, eta0_map, promise_width_h=0.5)

        # 输出 nodes_*_promise.csv（不覆盖原始数据集）
        try:
            promise_nodes_path = ut._derive_promise_nodes_path(file_path)
            ut.write_promise_nodes_csv(file_path, promise_nodes_path, eta0_map, promise_width_h=0.5)
            if verbose:
                print(f"[PROMISE] wrote: {promise_nodes_path}")
        except Exception as _e:
            print(f"[PROMISE-WARN] 写出 promise nodes 失败: {_e}")

    # 统一口径：全系统完成时刻（卡车到达客户/基站 + 无人机完成）
    finish_all_times = dict(arrival_times)
    finish_all_times.update(finish_times)
    system_finish_time = max(total_time, max(base_finish_times.values()) if base_finish_times else 0.0)

    sim.check_disjoint(data, best_route, best_b2d)
    if verbose:
        print("最优卡车路径（按 NODE_ID）:", [data.nodes[i]['node_id'] for i in best_route])
        print(f"最终: 成本={best_cost:.3f}, 卡车距={best_truck_dist:.3f}, "
              f"无人机距={best_drone_dist:.3f}, 总迟到={best_total_late:.3f}, "
              f"卡车总时间={best_truck_time:.2f}h, 系统完成时间={system_finish_time:.2f}h")
        print("各基站完成时间：")
        for b, t_fin in base_finish_times.items():
            n = data.nodes[b]
            print(f"  base node_id={n['node_id']}, type={n['node_type']}, 完成时间={t_fin:.2f}h")

    if enable_plot:
        visualize_truck_drone(data, best_route, best_b2d, title="Scenario 0: original (no relocation)")

    # ===================== 4) 结果表：先记场景0（FULL口径）=====================
    scenario_results = []
    full_eval0 = sim.evaluate_full_system(
        data, best_route, best_b2d,
        alpha_drone=0.3, lambda_late=0.0,
        truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
    )

    # [PROMISE] 场景0：输出 late_prom/late_eff（late_eff 以冻结窗为准）
    _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
    ut.emit_scene_late_logs(_late_dir, scene_idx=0, decision_time=0.0, data=data, full_route=best_route, full_b2d=best_b2d, full_eval=full_eval0, prefix="", drone_speed=sim.DRONE_SPEED_UNITS)
    # 中文注释：scene=0（初始静态解）也输出迟到分解
    if DEBUG_LATE and ((DEBUG_LATE_SCENES is None) or (0 in DEBUG_LATE_SCENES)):
        # 中文注释：debug_print_lateness_topk 已在 slim 版本中移除（避免控制台大输出拖慢实验）。
        # 如需查看 TopK 迟到客户，请查 late_logs/*.csv（emit_scene_late_logs 会写出）。
        pass

    scenario_results.append(ut._pack_scene_record(0, 0.0, full_eval0, num_req=0, num_acc=0, num_rej=0, alpha_drone=0.3, lambda_late=50.0))
    global_xlim, global_ylim = compute_global_xlim_ylim(
        data=data,
        reloc_radius=ab_cfg.get("reloc_radius", 0.8),
        pad_min=5.0,
        step_align=10.0
    )

    # ===================== 5) 动态循环初始化（“全局完整口径”状态）=====================
    if perturbation_times:
        data_cur = data

        full_route_cur = best_route.copy()
        full_b2d_cur = {b: cs.copy() for b, cs in best_b2d.items()}

        full_arrival_cur = arrival_times  # 全局从0开始
        full_depart_cur = depart_times  # 全局从0开始
        full_finish_cur = finish_all_times  # 全局从0开始（包含卡车+无人机完成时刻）

        scene_idx = 1
        t_prev = 0.0  # 上一决策时刻（只用来取“请求窗口”）
        # ---------- 5.0 动态请求流：仅由 events.csv 驱动（不在求解阶段随机生成请求） ----------
        reloc_radius = float(ab_cfg.get("reloc_radius", 0.8)) if ab_cfg else 0.8
        if offline_groups is None:
            raise RuntimeError("动态模式需要提供 events_path（events.csv），以保证请求流可复现且公平")
        if verbose:
            print(f"[RELOC-PLAN] mode=events radius={reloc_radius}")


        decision_times_list = [float(x) for x in perturbation_times]
        for _k_dec, decision_time in enumerate(decision_times_list):

            # ---------- 决策时刻护栏：不得逆序 ----------
            if decision_time < t_prev - 1e-9:
                raise RuntimeError(f"[GUARD] 决策时刻逆序：t_prev={t_prev}, t_dec={decision_time}")

            print(f"\n===== 场景 {scene_idx}: 决策时刻 t = {decision_time:.2f} h 应用位置变更后 =====")
            if seed is not None:
                ut.set_seed(int(seed) + int(scene_idx))

            # ---------- 5.1 split：在决策点前已服务/未服务 + 虚拟位置 ----------
            served_nodes, remaining_nodes, current_node, virtual_pos, prefix_route = dyn.split_route_by_decision_time(
                full_route_cur,
                full_arrival_cur,
                decision_time,
                data_cur.central_idx,
                data_cur
            )
            prefix_route_for_plot = prefix_route[:]

            served_nids = [data_cur.nodes[i]['node_id'] for i in served_nodes]
            remaining_nids = [data_cur.nodes[i]['node_id'] for i in remaining_nodes]
            if verbose:
                print(f"    已服务(卡车路径客户) NODE_ID: {served_nids}")
                print(f"    未服务(卡车路径客户) NODE_ID: {remaining_nids}")

            drone_served = dyn.get_drone_served_before_t(full_b2d_cur, full_finish_cur, decision_time)
            drone_served_uniq = sorted(set(drone_served))
            if verbose:
                print("    已服务(无人机) NODE_ID:", [data_cur.nodes[i]['node_id'] for i in drone_served_uniq])

            all_customers = {i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'customer'}

            # 【关键修复】卡车已服务集合必须以 split 的 served_nodes 为准，避免与 remaining_nodes 冲突导致漏点
            truck_served = set(served_nodes)

            drone_served_set = set(dyn.get_drone_served_before_t(full_b2d_cur, full_finish_cur, decision_time))
            served_all = truck_served | drone_served_set
            unserved_all = all_customers - served_all
            # 自检：split 给出的 served/remaining 必须互斥（否则时间轴有bug）
            _inter = set(served_nodes) & set(remaining_nodes)
            if _inter:
                raise RuntimeError(f"[BUG][SPLIT] served_nodes 与 remaining_nodes 重叠: {_inter}")

            if verbose:
                print(
                    f"    已完成(全系统) 客户数: {len(served_all)} = truck {len(truck_served)} + drone {len(drone_served_set)} (去重后)")
                print(f"    未完成(全系统) 客户数: {len(unserved_all)}")
                print("    未完成(全系统) NODE_ID:", [data_cur.nodes[i]['node_id'] for i in sorted(unserved_all)])

            # ---------- 5.2 基站集合（语义区分：future vs feasible） ----------
            all_bases = [i for i, n in enumerate(data_cur.nodes) if n.get('node_type') == 'base']
            if data_cur.central_idx not in all_bases:
                all_bases.append(data_cur.central_idx)
            route_set_cur = set(full_route_cur)
            bases_in_plan = [b for b in all_bases if (b in route_set_cur) or (b == data_cur.central_idx)]
            visited_bases = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) <= decision_time + 1e-9]
            bases_to_visit = [b for b in bases_in_plan if full_arrival_cur.get(b, float('inf')) > decision_time + 1e-9]
            feasible_bases_for_drone = sorted(set(visited_bases + bases_to_visit))
            arrival_prefix = {b: full_arrival_cur[b] for b in visited_bases if b in full_arrival_cur}
            if verbose:
                print(f'    [bases_to_visit] t={decision_time:.2f}h => {[data_cur.nodes[b]["node_id"] for b in bases_to_visit]}')

            # ---------- 5.3 应用位置变更（events.csv：仅事实；ACCEPT/REJECT 由算法决定） ----------
            client_to_base_cur = dyn.build_client_to_base_map(full_b2d_cur)

            req_override = []
            predefined_xy = {}
            predefined_types = {}
            predefined_delta_avail = {}
            decisions_raw = []
            req_clients = []

            key = round(float(decision_time), 6)
            evs = offline_groups.get(key, []) if offline_groups is not None else []
            _ev_meta = {}
            for e in evs:
                nid = int(e.get('NODE_ID', 0))
                cidx = nodeid2idx.get(nid, None)
                if cidx is None:
                    continue
                # served_all 在本轮 5.1 后已计算
                if 'served_all' in locals() and (int(cidx) in served_all):
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
                        'APPLIED_X': '',
                        'APPLIED_Y': '',
                        'FORCE_TRUCK': '',
                        'BASE_LOCK': ''
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

            # 去重（保持稳定顺序）
            seen = set()
            req_override = [c for c in req_override if (c not in seen and not seen.add(c))]

            if verbose:
                print(f"    [REQ-OFFLINE] t={decision_time:.2f}h events={len(evs)} active={len(req_override)}")

            if len(req_override) > 0:
                # 将本决策点事件 meta 暂存到 data_cur（仅用于落盘 decision_log_rows）
                data_cur._offline_ev_meta = _ev_meta
                data_prelim, decisions_raw, req_clients = dyn.apply_relocations_for_decision_time(
                    data_cur, t_prev, decision_time,
                    full_depart_cur, full_finish_cur, full_arrival_cur,
                    client_to_base_cur,
                    req_override, predefined_xy, predefined_types, predefined_delta_avail
                )

                # ---------- [OFFLINE] 将本决策点的 ACCEPT/REJECT 结果写入 decision_log_rows ----------
                _meta = {}
                try:
                    _meta = getattr(data_cur, "_offline_ev_meta", {}) or {}
                except Exception:
                    _meta = {}
                if False and _meta and decisions_raw:
                    for _tup in decisions_raw:
                        try:
                            _cidx, _nid, _dec, _reason, _ox, _oy, _nx, _ny = _tup[:8]
                        except Exception:
                            continue
                        _m = _meta.get(int(_cidx), {})
                        _eid = _m.get("EVENT_ID", "")
                        _ecls = _m.get("EVENT_CLASS", "")
                        _apx = _nx if str(_dec).upper() == "ACCEPT" else _ox
                        _apy = _ny if str(_dec).upper() == "ACCEPT" else _oy
                        decision_log_rows.append({
                            "EVENT_ID": _eid,
                            "EVENT_TIME": float(decision_time),
                            "NODE_ID": int(_nid),
                            "DECISION": str(_dec),
                            "REASON": str(_reason),
                            "OLD_X": float(_ox),
                            "OLD_Y": float(_oy),
                            "NEW_X": float(_nx),
                            "NEW_Y": float(_ny),
                            "EVENT_CLASS": str(_ecls),
                            "APPLIED_X": float(_apx),
                            "APPLIED_Y": float(_apy),
                            "FORCE_TRUCK": "",
                            "BASE_LOCK": ""
                        })
                # 清空，避免跨决策点污染
                try:
                    data_cur._offline_ev_meta = {}
                except Exception:
                    pass
            else:
                data_prelim = data_cur

            # ---------- 5.4 判断是否存在尚未完成的无人机任务（用于 early-stop 口径） ----------
            unfinished_drone_exist = any(
                (full_finish_cur.get(_c, float('inf')) > decision_time + 1e-9) for _c in list(full_finish_cur.keys())
            )
            if (len(unserved_all) == 0) and (not unfinished_drone_exist) and (len(req_clients) == 0):
                if verbose:
                    print(
                        f"    [EARLY-STOP] t={decision_time:.2f}h：无未服务客户、无未完成无人机任务、且无新请求，提前结束后续决策点。")
                break
            # ---------- 决策点无新请求：不进行路径重规划（继续沿用原计划，等待后续决策点） ----------
            if len(req_clients) == 0:
                if verbose:
                    print(f"    [SKIP-REPLAN] t={decision_time:.2f}h：本决策点无新请求，不触发重规划。")
                t_prev = decision_time
                scene_idx += 1
                continue

            for d in decisions_raw:

                if verbose:
                    print("  ", d)
            # ===== 2.1) 快速筛选：把“代价太大”的 ACCEPT 改为 REJECT =====
            DELTA_COST_MAX = 30.0
            DELTA_LATE_MAX = 0.10

            data_next, decisions, qf_deltas = dyn.quick_filter_relocations(
                data_cur=data_cur,
                data_prelim=data_prelim,
                full_route_cur=full_route_cur,  # 注意：用当前场景“完整计划”作为快速评估基线
                full_b2d_cur=full_b2d_cur,
                prefix_route=prefix_route,
                req_clients=req_clients,
                decisions=decisions_raw,
                alpha_drone=0.3,
                lambda_late=50.0,
                truck_speed=sim.TRUCK_SPEED_UNITS,
                drone_speed=sim.DRONE_SPEED_UNITS,
                delta_cost_max=DELTA_COST_MAX,
                delta_late_max=DELTA_LATE_MAX,
            )

            # ---------- [OFFLINE] 将最终 ACCEPT/REJECT（含Δ指标与承诺/有效截止时间）写入 decision_log_rows ----------
            try:
                for (cid, dec, nx, ny, reason) in decisions:
                    cid = int(cid)
                    meta = _ev_meta.get(cid, {}) if isinstance(_ev_meta, dict) else {}
                    _eid = int(meta.get('EVENT_ID', -1))
                    _ecls = str(meta.get('EVENT_CLASS', ''))
                    _dav = float(meta.get('DELTA_AVAIL_H', predefined_delta_avail.get(cid, 0.0) if 'predefined_delta_avail' in locals() else 0.0))

                    # old 坐标来自 data_cur（apply 使用深拷贝，不会污染 data_cur）
                    _ox = float(data_cur.nodes[cid].get('x', 0.0))
                    _oy = float(data_cur.nodes[cid].get('y', 0.0))

                    # applied 坐标来自 data_next（quick_filter 最终状态）
                    _apx = float(data_next.nodes[cid].get('x', _ox))
                    _apy = float(data_next.nodes[cid].get('y', _oy))

                    pr = float(data_next.nodes[cid].get('prom_ready', data_next.nodes[cid].get('ready_time', 0.0)))
                    pd = float(data_next.nodes[cid].get('prom_due', data_next.nodes[cid].get('due_time', 0.0)))
                    ed = float(data_next.nodes[cid].get('due_time', pd))

                    drec = qf_deltas.get(cid, {}) if isinstance(qf_deltas, dict) else {}
                    d_cost = float(drec.get('D_COST', 0.0))
                    d_lp = float(drec.get('D_LATE_PROM', 0.0))
                    d_le = float(drec.get('D_LATE_EFF', 0.0))

                    decision_log_rows.append({
                        'EVENT_ID': _eid,
                        'EVENT_TIME': float(decision_time),
                        'NODE_ID': int(data_cur.nodes[cid].get('node_id', cid)),
                        'DECISION': str(dec),
                        'REASON': str(reason),
                        'OLD_X': _ox,
                        'OLD_Y': _oy,
                        'NEW_X': float(nx),
                        'NEW_Y': float(ny),
                        'EVENT_CLASS': str(_ecls),
                        'APPLIED_X': _apx,
                        'APPLIED_Y': _apy,
                        'FORCE_TRUCK': int(data_next.nodes[cid].get('force_truck', 0)),
                        'BASE_LOCK': ('' if data_next.nodes[cid].get('base_lock', None) is None else int(data_next.nodes[cid].get('base_lock'))),
                        'DELTA_AVAIL_H': _dav,
                        'PROM_READY': pr,
                        'PROM_DUE': pd,
                        'EFFECTIVE_DUE': ed,
                        'D_COST': d_cost,
                        'D_LATE_PROM': d_lp,
                        'D_LATE_EFF': d_le,
                    })
            except Exception as _e:
                print(f"[OFFLINE-LOG-WARN] write decision_log_rows failed: {_e}")



            # 兜底强制卡车直送集合（关键：先算它，后面 allowed 要用）
            forced_truck = {i for i in data_next.customer_indices
                            if data_next.nodes[i].get('force_truck', 0) == 1}
            # req_set：只让“未完成客户”进入后续重规划集合（关键修复）
            req_all = {int(c) for c in req_clients}
            req_set = {c for c in req_all if c in unserved_all}  # 只保留未完成
            req_served = req_all - req_set  # 已完成但提出请求的客户

            if req_served:
                print("[GUARD] 已完成客户提出变更请求(将不参与重规划):",
                      [data_next.nodes[i]["node_id"] for i in sorted(req_served)])

            print("本次决策的客户变更结果：")
            if not decisions:
                print("  无客户在该时刻提出变更请求，或全部不满足条件。")
            else:
                for it in decisions:
                    # quick_filter 后 decisions 为 (cid, dec, nx, ny, reason)
                    if isinstance(it, (list, tuple)) and len(it) == 8:
                        c, nid, dec, reason, ox, oy, nx, ny = it
                        print(f"  client_idx={c}, node_id={nid}, decision={dec}, "
                              f"old=({ox:.2f},{oy:.2f}), new=({nx:.2f},{ny:.2f}), 原因: {reason}")
                    elif isinstance(it, (list, tuple)) and len(it) == 5:
                        c, dec, nx, ny, reason = it
                        nid = data_cur.nodes[int(c)]['node_id']
                        ox = data_cur.nodes[int(c)]['x']
                        oy = data_cur.nodes[int(c)]['y']
                        print(f"  client_idx={c}, node_id={nid}, decision={dec}, "
                              f"old=({ox:.2f},{oy:.2f}), new=({nx:.2f},{ny:.2f}), 原因: {reason}")
                    else:
                        print(f"  [WARN] decisions 格式异常: {it}")

            # ---------- 护栏：无有效变更时跳过重规划（只推进时间窗口） ----------
            # 说明：如果本决策窗口内没有任何 ACCEPT（即没有坐标/模式变更落地），且没有新增 force_truck，
            # 则继续沿用 full_route_cur/full_b2d_cur，不做 classify + ALNS（节省算力，且更符合“无事件不重规划”）。
            num_req = len(decisions)
            num_acc = sum(1 for d in decisions if dyn._decision_tag(d).startswith("ACCEPT"))
            num_rej = sum(1 for d in decisions if dyn._decision_tag(d).startswith("REJECT"))
            if (num_acc == 0) and (len(forced_truck) == 0):
                if verbose:
                    print(f"    [SKIP-REPLAN] t={decision_time:.2f}h：无ACCEPT请求且无force_truck变更，跳过路径重规划，沿用当前全局计划。")
                # 仍然记录该场景指标（全局口径，不重规划）
                full_eval_skip = sim.evaluate_full_system(
                    data_next, full_route_cur, full_b2d_cur,
                    alpha_drone=0.3, lambda_late=50.0,
                    truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
                )
                scenario_results.append(
                    dyn._pack_scene_record(scene_idx, decision_time, full_eval_skip,
                                       num_req=num_req, num_acc=num_acc, num_rej=num_rej,
                                       alpha_drone=0.3, lambda_late=50.0)
                )

                # [PROMISE] 场景日志输出（skip-replan）
                _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
                ut.emit_scene_late_logs(_late_dir, scene_idx=scene_idx, decision_time=decision_time, data=data_next, full_route=full_route_cur, full_b2d=full_b2d_cur, full_eval=full_eval_skip, prefix="", drone_speed=sim.DRONE_SPEED_UNITS)
                # 推进窗口，进入下一决策点（不改变 full_* 计划）
                data_cur = data_next
                t_prev = decision_time
                scene_idx += 1
                continue

            # ---------- 5.4 重规划起点（虚拟节点） ----------
            if virtual_pos is not None:
                start_idx_for_alns = dyn.add_virtual_truck_position_node(data_next, virtual_pos)
            else:
                start_idx_for_alns = current_node

            # 兜底修正：避免从仓库 0 重新起步（注意：这里绝不覆盖 t_prev）
            if (start_idx_for_alns == data_next.central_idx
                    and decision_time > 1e-6
                    and len(remaining_nodes) > 0):

                first_node = remaining_nodes[0]
                seg_t0 = full_arrival_cur.get(data_cur.central_idx, 0.0)
                seg_t1 = full_arrival_cur.get(first_node, seg_t0 + 1e-6)

                if seg_t1 > seg_t0 + 1e-9:
                    ratio = (decision_time - seg_t0) / (seg_t1 - seg_t0)
                    ratio = max(0.0, min(1.0, ratio))

                    x0 = data_next.nodes[data_next.central_idx]['x']
                    y0 = data_next.nodes[data_next.central_idx]['y']
                    x1 = data_next.nodes[first_node]['x']
                    y1 = data_next.nodes[first_node]['y']

                    x_cur = x0 + ratio * (x1 - x0)
                    y_cur = y0 + ratio * (y1 - y0)

                    start_idx_for_alns = dyn.add_virtual_truck_position_node(data_next, (x_cur, y_cur))
                    dprint(f"  [fix] 起点从仓库 0 修正为虚拟位置 idx={start_idx_for_alns}, pos=({x_cur:.2f},{y_cur:.2f})")

            # ---------- 5.5 构造 allowed_customers（这是你之前最容易乱的地方） ----------
            remaining_truck_route_nodes = set(remaining_nodes)

            unfinished_drone = set()
            for b, clients in full_b2d_cur.items():
                t_base = full_arrival_cur.get(b, float('inf'))
                for c in clients:
                    if full_finish_cur.get(c, float('inf')) <= decision_time + 1e-9:
                        continue
                    if t_base > decision_time + 1e-9:
                        unfinished_drone.add(c)
                    else:
                        if full_depart_cur.get(c, float('inf')) > decision_time + 1e-9:
                            unfinished_drone.add(c)
            # ---------- 修复：force_truck 只对“未完成客户”生效 ----------
            forced_truck = set(forced_truck)

            # 1) 已服务的 force_truck 必须剔除，否则与 served_guard 冲突
            served_in_force = forced_truck & served_all
            if served_in_force:
                if verbose:
                    print(f"[FIX] force_truck 中包含已服务客户，已剔除：{sorted(served_in_force)}")
                forced_truck -= served_in_force
            # ====== 强制对齐：卡车 served/remaining 以 split 输出为唯一真值（放在 allowed_customers 构造前）======
            truck_served = set(served_nodes)  # t_dec 前卡车已服务客户集合
            remaining_truck_route_nodes = set(remaining_nodes)  # t_dec 后卡车后缀仍需考虑的客户集合

            # 自检：二者必须互斥，否则后续会出现“本该重规划却被当 served 剔除”的漏点
            _inter = truck_served & remaining_truck_route_nodes
            if _inter:
                if verbose:
                    print("[BUG][SPLIT] truck_served 与 remaining 重叠：", sorted(_inter))
                raise RuntimeError("[BUG][SPLIT] served/remaining 不互斥，请检查 split 逻辑或时间轴")

            # 2) 用过滤后的 forced_truck 构造 allowed
            allowed_customers = (remaining_truck_route_nodes | unfinished_drone | req_set | forced_truck)

            # 护栏：已完成客户绝不允许进入 allowed（否则会被重新规划/重复服务）
            served_in_allowed = allowed_customers & served_all
            if served_in_allowed:
                if verbose:
                    print(f"[GUARD][FIX] served_all ∩ allowed_customers 非空，已自动剔除： {sorted(served_in_allowed)}")
                allowed_customers -= served_in_allowed

            # 护栏：force_truck 必须进 allowed（这里 forced_truck 已经剔除了 served，所以不再矛盾）
            missing = forced_truck - allowed_customers
            if missing:
                raise RuntimeError(f"[GUARD] force_truck 客户未进入 allowed_customers: node_id={sorted(missing)}")

            # ====== 5.A 构造结构敏感集合 ======
            C_moved_accept, C_moved_reject = set(), set()
            for it in decisions:
                cid = int(it[0])
                if cid not in unserved_all:
                    continue  # 已完成客户不进入结构敏感集合（否则算子会“扰动已完成点”）
                dec = str(it[1])
                if dec.startswith("ACCEPT"):
                    C_moved_accept.add(cid)
                elif dec.startswith("REJECT"):
                    C_moved_reject.add(cid)

            C_force_truck = {i for i in data_next.customer_indices if data_next.nodes[i].get("force_truck", 0) == 1}

            # 覆盖边界：只在 allowed_customers 中挑 top-k
            def _boundary_score(i):
                xi, yi = data_next.nodes[i]["x"], data_next.nodes[i]["y"]
                best = float("inf")
                for b in feasible_bases_for_drone:
                    xb, yb = data_next.nodes[b]["x"], data_next.nodes[b]["y"]
                    d = ((xi - xb) ** 2 + (yi - yb) ** 2) ** 0.5
                    best = min(best, abs(2 * d - sim.DRONE_RANGE_UNITS))
                return best

            k_boundary = 6
            cand_boundary = [i for i in allowed_customers
                             if data_next.nodes[i].get("node_type") == "customer"
                             and i not in C_force_truck]
            cand_boundary.sort(key=_boundary_score)
            C_boundary = set(cand_boundary[:k_boundary])

            print(f"[SET] moved_acc={len(C_moved_accept)} moved_rej={len(C_moved_reject)} "
                  f"force={len(C_force_truck)} boundary={len(C_boundary)}")

            # ---------- 5.6 分类（只在 allowed 内重新分配 truck/drone） ----------
            base_to_drone_next, truck_next = sim.classify_clients_for_drone(
                data_next,
                allowed_customers=allowed_customers,
                feasible_bases=feasible_bases_for_drone
            )

            drone_assigned = {c for cs in base_to_drone_next.values() for c in cs}

            # 护栏：force_truck 不允许进无人机
            bad = forced_truck & drone_assigned
            if bad:
                raise RuntimeError(f"[GUARD] force_truck 客户被分配给无人机: node_id="
                                   f"{[data_next.nodes[i]['node_id'] for i in bad]}")

            # ---------- 5.8 ALNS 重规划（后缀） ----------
            ctx_for_alns = dict(ab_cfg)  # 关键：先注入消融配置（paired/free/算子池）
            ctx_for_alns["verbose"] = verbose
            ctx_for_alns.update({
                "C_moved_accept": C_moved_accept,
                "C_moved_reject": C_moved_reject,
                "C_force_truck": C_force_truck,
                "C_boundary": C_boundary,
                "min_remove": 3,  # 动态场景至少移除3个点，保证有“重构力度”
            })

            # 基站语义区分：用于无人机可行基站与到达时间合并（关键修复）
            ctx_for_alns["feasible_bases_for_drone"] = feasible_bases_for_drone
            ctx_for_alns["visited_bases"] = set(visited_bases)
            ctx_for_alns["arrival_prefix"] = arrival_prefix


            (route_next, b2d_next, cost_next, truck_dist_next, drone_dist_next, total_late_next,
             truck_time_next) = alns_truck_drone(data_next, base_to_drone_next, max_iter=1000,
                                                 remove_fraction=0.1, T_start=1.0, T_end=0.01, alpha_drone=0.3,
                                                 lambda_late=50.0,
                                                 truck_customers=truck_next, use_rl=False, rl_tau=0.5, rl_eta=0.1,
                                                 start_idx=start_idx_for_alns,
                                                 start_time=decision_time, bases_to_visit=bases_to_visit,
                                                 ctx=ctx_for_alns)

            if b2d_next is None:
                raise RuntimeError("alns_truck_drone 返回 b2d_next=None，请检查函数返回值。")

            # 本场景后缀时间表（从 decision_time 开始）
            arrival_next, total_time_next, _ = sim.compute_truck_schedule(
                data_next, route_next, start_time=decision_time, speed=sim.TRUCK_SPEED_UNITS
            )
            arrival_next_merged = dict(arrival_prefix)
            arrival_next_merged.update(arrival_next)
            depart_next, finish_next, base_finish_next = sim.compute_multi_drone_schedule(
                data_next, b2d_next, arrival_next_merged,
                num_drones_per_base=sim.NUM_DRONES_PER_BASE,
                drone_speed=sim.DRONE_SPEED_UNITS
            )
            system_finish_next = max(
                total_time_next,
                max(base_finish_next.values()) if base_finish_next else 0.0
            )
            sim.check_disjoint(data_next, route_next, b2d_next)

            print(f"场景 {scene_idx}: 成本={cost_next:.3f}, 卡车距={truck_dist_next:.3f}, "
                  f"无人机距={drone_dist_next:.3f}, 总迟到={total_late_next:.3f}, "
                  f"卡车总时间={truck_time_next:.2f}h, 系统完成时间={system_finish_next:.2f}h")

            num_req = len(decisions)
            num_acc = sum(1 for d in decisions if dyn._decision_tag(d).startswith("ACCEPT"))
            num_rej = sum(1 for d in decisions if dyn._decision_tag(d).startswith("REJECT"))

            # ---------- 5.9 拼接“全局完整解”（FULL） ----------
            suffix_for_full = route_next[:]
            if suffix_for_full and (
                    data_next.nodes[suffix_for_full[0]].get('node_type') == 'truck_pos'
                    or data_next.nodes[suffix_for_full[0]].get('node_id') == -1
            ):
                suffix_for_full = suffix_for_full[1:]

            full_route_next = dyn._merge_prefix_suffix(prefix_route, suffix_for_full)

            full_b2d_next = {b: cs.copy() for b, cs in full_b2d_cur.items()}

            # 从历史无人机任务中移除 allowed（避免重复）
            allowed_set = set(allowed_customers)
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in allowed_set]

            # 合并本场景新分配
            for b, cs in b2d_next.items():
                full_b2d_next.setdefault(b, []).extend(cs)

            # 再移除 force_truck（兜底）
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in forced_truck]

            # 全局去重：一客一基站
            seen = set()
            for b in list(full_b2d_next.keys()):
                new_list = []
                for c in full_b2d_next[b]:
                    if c in seen:
                        continue
                    seen.add(c)
                    new_list.append(c)
                full_b2d_next[b] = new_list

            # 卡车优先：去掉与卡车路径冲突的无人机客户
            truck_nodes_set = {
                i for i in full_route_next
                if 0 <= i < len(data_next.nodes) and data_next.nodes[i].get('node_type') == 'customer'
            }
            for b in list(full_b2d_next.keys()):
                full_b2d_next[b] = [c for c in full_b2d_next[b] if c not in truck_nodes_set]
            full_route_next = cover_uncovered_by_truck_suffix(
                data_next, full_route_next, full_b2d_next, prefix_len=len(prefix_route), verbose=verbose
            )
            # 如果你有 guardrail_check 就保留，没有就删
            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" after-merge scene={scene_idx}")

            # FULL 全局时间轴重算（从0开始）
            full_arrival_next, full_total_time_next, full_truck_late_next = sim.compute_truck_schedule(
                data_next, full_route_next, start_time=0.0, speed=sim.TRUCK_SPEED_UNITS
            )
            full_depart_next, full_finish_next, full_base_finish_next = sim.compute_multi_drone_schedule(
                data_next, full_b2d_next, full_arrival_next,
                num_drones_per_base=sim.NUM_DRONES_PER_BASE,
                drone_speed=sim.DRONE_SPEED_UNITS
            )
            # ---------- 统一 finish_time 口径：卡车客户=truck_arrival，无人机客户=drone_finish ----------
            full_finish_all_next = dict(full_arrival_next)  # 先用卡车到达时间填满（包含所有卡车路径节点）

            # 用无人机完成时间覆盖对应客户（只覆盖 customer 节点）
            for _cid, _fin in full_finish_next.items():
                try:
                    _cid_int = int(_cid)
                except Exception:
                    continue
                if 0 <= _cid_int < len(data_next.nodes) and str(
                        data_next.nodes[_cid_int].get("node_type", "")).lower() == "customer":
                    full_finish_all_next[_cid_int] = float(_fin)

            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" after-schedule scene={scene_idx}")

            full_eval = sim.evaluate_full_system(
                data_next, full_route_next, full_b2d_next,
                alpha_drone=0.3, lambda_late=50.0,
                truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
            )

            # [PROMISE] 场景日志输出（replan）
            _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
            ut.emit_scene_late_logs(_late_dir, scene_idx=scene_idx, decision_time=decision_time, data=data_next, full_route=full_route_next, full_b2d=full_b2d_next, full_eval=full_eval, prefix="", drone_speed=sim.DRONE_SPEED_UNITS)

            check_info = sim.sanity_check_full(data_next, full_route_next, full_b2d_next)
            print("[FULL-check]", check_info)
            print(f"[FULL] cost={full_eval['cost']:.3f}, system_time={full_eval['system_time']:.3f}h, "
                  f"late={full_eval['total_late']:.3f} (truck={full_eval['truck_late']:.3f}, drone={full_eval['drone_late']:.3f}), "
                  f"truck_dist={full_eval['truck_dist']:.3f}, drone_dist={full_eval['drone_dist']:.3f}")

            scenario_results.append({
                "scene": scene_idx,
                "t_dec": decision_time,
                "cost": full_eval["cost"],
                "base_cost": full_eval.get("truck_dist_eff", full_eval["truck_dist"]) + 0.3 * full_eval["drone_dist"],
                "penalty": 50.0 * full_eval["total_late"],
                "lambda_late": 50.0,
                "truck_dist": full_eval["truck_dist"],
                "drone_dist": full_eval["drone_dist"],
                "system_time": full_eval["system_time"],
                "truck_late": full_eval["truck_late"],
                "drone_late": full_eval["drone_late"],
                "total_late": full_eval["total_late"],
                "num_req": num_req,
                "num_acc": num_acc,
                "num_rej": num_rej
            })

            # 画图：对比前一轮无人机集合（可选）
            drone_set_before = set()
            for cs in full_b2d_cur.values():
                drone_set_before.update(cs)

            if "guardrail_check" in globals():
                sim.guardrail_check(data_next, full_route_next, full_b2d_next, tag=f" before-plot scene={scene_idx}")

                        # 兼容：visualize_truck_drone 需要 8 元组 (cidx, nid, dec, reason, ox, oy, nx, ny)
            decisions_viz = _normalize_decisions_for_viz(data_cur, decisions)
            if enable_plot:
                visualize_truck_drone(
                    data_next,
                    full_route_next,
                    full_b2d_next,
                    title=f"Scenario {scene_idx}: progress view (t={decision_time:.2f}h)",
                    xlim=global_xlim,
                    ylim=global_ylim,
                    decision_time=decision_time,
                    truck_arrival=full_arrival_next,
                    drone_finish=full_finish_next,
                    prefix_route=prefix_route_for_plot,
                    virtual_pos=virtual_pos,
                    relocation_decisions=decisions_viz,
                    drone_set_before=drone_set_before
                )

            # ---------- 5.10 更新滚动状态，进入下一决策点 ----------
            data_cur = data_next
            full_route_cur = full_route_next
            full_b2d_cur = full_b2d_next
            full_arrival_cur = full_arrival_next
            full_depart_cur = full_depart_next
            full_finish_cur = full_finish_all_next

            t_prev = decision_time
            scene_idx += 1

    # ===================== 6) 汇总输出（run_one 里建议不打印，交给 main 或 ablation）=====================

    # ---------- 保存 decision_log（离线 events.csv 模式） ----------
    try:
        if offline_groups is not None:
            _out = decision_log_path
            if (not _out) or (str(_out).strip() == ""):
                base = os.path.splitext(os.path.basename(file_path))[0]
                _dir = os.path.dirname(events_path) if events_path else os.path.dirname(file_path)
                _out = os.path.join(_dir or ".", f"decision_log_{base}_seed{seed}.csv")
            ut.save_decision_log(decision_log_rows, _out)
            if verbose:
                print(f"[LOG] decision log saved: {_out} (rows={len(decision_log_rows)})")
    except Exception as _e:
        if verbose:
            print("[WARN] decision_log 保存失败：", _e)

    return scenario_results

def _resolve_operator_list(names, g):
    """把字符串函数名解析成函数对象；先找当前 globals，再找 ops 模块。"""
    ops_list = []
    for nm in names:
        # 1. 先尝试在当前文件查找（兼容旧写法）
        fn = g.get(nm, None)
        # 2. 如果没找到，去 operators 模块找
        if fn is None and "ops" in g:
            fn = getattr(g["ops"], nm, None)

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
            new_cfg["DESTROYS"] = _resolve_operator_list(new_cfg["DESTROYS"], globals())

    if "REPAIRS" in new_cfg and new_cfg["REPAIRS"]:
        if isinstance(new_cfg["REPAIRS"][0], str):
            new_cfg["REPAIRS"] = _resolve_operator_list(new_cfg["REPAIRS"], globals())

    # ALLOWED_PAIRS（paired 模式）
    if "ALLOWED_PAIRS" in new_cfg and new_cfg["ALLOWED_PAIRS"]:
        pairs = []
        for dnm, rnm in new_cfg["ALLOWED_PAIRS"]:
            # destroy
            if isinstance(dnm, str):
                D = globals().get(dnm) or getattr(ops, dnm, None)
                if D is None or not callable(D):
                    raise RuntimeError(f"[CFG] 未找到 destroy: {dnm}")
            elif callable(dnm):
                D = dnm
            else:
                raise RuntimeError(f"[CFG] destroy 既不是字符串也不是函数对象: {dnm}")

            # repair
            if isinstance(rnm, str):
                R = globals().get(rnm) or getattr(ops, rnm, None)
                if R is None or not callable(R):
                    raise RuntimeError(f"[CFG] 未找到 repair: {rnm}")
            elif callable(rnm):
                R = rnm
            else:
                raise RuntimeError(f"[CFG] repair 既不是字符串也不是函数对象: {rnm}")

            pairs.append((D, R))
        new_cfg["ALLOWED_PAIRS"] = pairs

    return new_cfg

def _normalize_perturbation_times(times):
    """规范化决策时刻列表：
    - 过滤 <=0 的时刻（t=0 的初始场景由系统自动包含，避免重复）
    - 去重（按 1e-6 精度）
    - 升序排序
    """
    import math
    if not times:
        return []
    cleaned = []
    for t in times:
        try:
            ft = float(t)
        except Exception:
            continue
        if math.isnan(ft) or math.isinf(ft):
            continue
        if ft <= 1e-9:
            continue
        cleaned.append(round(ft, 6))
    return sorted(set(cleaned))

def run_static_truck_only(
        file_path: str,
        seed: int,
        ab_cfg: dict,
        enable_plot: bool = False,
        verbose: bool = True
):
    """静态-纯卡车基线（不使用无人机，不强制访问基站）。

    目的：和混合模式在“同一 ALNS 框架”下公平对比。
    做法：
    - 把所有 customer 标记 force_truck=1
    - base_to_drone_customers 置空
    - bases_to_visit 仅包含 central（避免初始路径把所有基站当作必访点）
    """
    if seed is not None:
        ut.set_seed(int(seed))

    ab_cfg = build_ab_cfg(ab_cfg)

    data = read_data(file_path, scenario=0, strict_schema=True)

    # 1) 强制所有客户都走卡车
    truck_customers = list(getattr(data, "customer_indices", []))
    for cid in truck_customers:
        try:
            data.nodes[cid]["force_truck"] = 1
        except Exception:
            pass

    base_to_drone_customers = {}  # 纯卡车：无人机任务为空

    ctx0 = dict(ab_cfg)
    ctx0["verbose"] = verbose

    (best_route,
     best_b2d,
     _best_cost_internal,
     _best_truck_dist,
     _best_drone_dist,
     _best_total_late,
     _best_truck_time) = alns_truck_drone(
        data,
        base_to_drone_customers,
        max_iter=1000,
        remove_fraction=0.1,
        T_start=1.0,
        T_end=0.01,
        alpha_drone=0.3,
        lambda_late=50.0,
        truck_customers=truck_customers,
        use_rl=False,
        rl_tau=0.5,
        rl_eta=0.1,
        bases_to_visit=[data.central_idx],   # 关键：不强制访问所有基站
        ctx=ctx0
    )

    # 二次兜底：确保无人机为空、force_truck 不被破坏
    best_route, best_b2d = sim.enforce_force_truck_solution(data, best_route, best_b2d)

    full_eval = sim.evaluate_full_system(
        data, best_route, best_b2d,
        alpha_drone=0.3, lambda_late=50.0,
        truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
    )

    rec = {
        "scene": 0,
        "t_dec": 0.0,
        "cost": full_eval["cost"],
        "base_cost": full_eval.get("truck_dist_eff", full_eval["truck_dist"]) + 0.3 * full_eval["drone_dist"],
        "penalty": 50.0 * full_eval["total_late"],
        "lambda_late": 50.0,
        "truck_dist": full_eval["truck_dist"],
        "drone_dist": full_eval["drone_dist"],
        "system_time": full_eval["system_time"],
        "truck_late": full_eval["truck_late"],
        "drone_late": full_eval["drone_late"],
        "total_late": full_eval["total_late"],
        "num_req": 0,
        "num_acc": 0,
        "num_rej": 0
    }

    if verbose:
        print("[TRUCK-ONLY] route NODE_ID:", [data.nodes[i].get("node_id", i) for i in best_route])
        print(f"[TRUCK-ONLY] cost={rec['cost']:.3f}, system_time={rec['system_time']:.3f}h, "
              f"late={rec['total_late']:.3f}, truck_dist={rec['truck_dist']:.3f}")

    if enable_plot:
        visualize_truck_drone(
            data, best_route, best_b2d,
            title="Static Truck-Only Baseline",
            show_base_radius=True,
            decisions=[],
            save_path=""
        )

    return rec

def main():
    """
    中文注释：主入口（不再使用命令行参数，所有实验参数集中在此处配置）。
    你只需要改下面这些变量即可复现：
    - file_path / events_path
    - seed / cfg / perturbation_times
    - road_factor（路况系数：只影响卡车弧距离=欧氏×系数，从而影响卡车时间/迟到与卡车距离成本）
    """
    print("[BOOT]", __file__, "DEBUG_LATE=", DEBUG_LATE, "DEBUG_LATE_SCENES=", DEBUG_LATE_SCENES)

    # ===== 1) 实验输入 =====
    file_path = r"D:\代码\ALNS+DL\nodes_25_seed2023_20260110_201842.csv"
    events_path = r"D:\代码\ALNS+DL\events_25_seed2023_20260110_201842.csv"  # 可选：离线事件脚本（仅事实，不含 accept/reject），为空则不使用
    seed = 2025
    cfg = CFG_D

    # 动态模式：决策点（小时），t=0 场景系统自动包含
    perturbation_times = [1.0, 2.0]

    enable_plot = True
    verbose = True

    # ===== 2) 路况系数（唯一入口：只放大卡车距离，不改速度）=====
    # 初始化仿真参数
    road_factor = 1.5
    sim.set_simulation_params(road_factor=road_factor)
    # 并且建议定义本地快捷变量，如果下面有用到
    TRUCK_SPEED_UNITS = sim.get_simulation_params()["TRUCK_SPEED_UNITS"]
    DRONE_SPEED_UNITS = sim.get_simulation_params()["DRONE_SPEED_UNITS"]
    NUM_DRONES_PER_BASE = sim.get_simulation_params()["NUM_DRONES_PER_BASE"]
    print(f"[PARAM] TRUCK_ROAD_FACTOR={sim.TRUCK_ROAD_FACTOR:.3f}; TRUCK_SPEED_UNITS={TRUCK_SPEED_UNITS:.3f} units/h (fixed); truck_arc = euclid * {sim.TRUCK_ROAD_FACTOR:.3f}")

    # ===== 3) 运行模式开关 =====
    # 3.1 静态对比：纯卡车 vs 混合（只跑 scene=0，不跑动态扰动）
    RUN_STATIC_COMPARE = False

    # 3.2 批量对比：多 seed + 多 CFG（使用同一 events.csv 即可保证同一请求流）
    RUN_BATCH = False

    # 3.3 最小可复现实验：road_factor 1.0 vs 1.5 应显著改变 system_time / late
    RUN_ROAD_SANITY = False

    # ===== 4) road sanity =====
    if RUN_ROAD_SANITY:
        for rf in [1.0, 1.5]:
            sim.set_simulation_params(rf)
            results = run_one(
                file_path=file_path, seed=seed, ab_cfg=cfg,
                perturbation_times=perturbation_times,
                enable_plot=False, verbose=False,
                events_path=events_path, decision_log_path=''
            )
            r0 = results[0]
            print(f"[SANITY] rf={rf:.1f} speed_units={TRUCK_SPEED_UNITS:.3f} truck_dist={r0['truck_dist']:.3f} drone_dist={r0['drone_dist']:.3f} system_time={r0['system_time']:.3f} total_late={r0['total_late']:.3f} cost={r0['cost']:.3f}")
        return

    # ===== 5) 静态纯卡车 vs 混合对比 =====
    if RUN_STATIC_COMPARE:
        # 只跑静态场景：不加任何决策点
        pert_static = []

        # 5.1 混合模式（原 run_one）
        sim.set_simulation_params(road_factor=road_factor)
        res_mixed = run_one(
            file_path=file_path, seed=seed, ab_cfg=cfg,
            perturbation_times=pert_static,
            enable_plot=False, verbose=False,
            events_path=events_path, decision_log_path=''
        )[0]

        # 5.2 纯卡车（不访问基站、不派无人机）
        sim.set_simulation_params(road_factor=road_factor)
        res_truck = run_static_truck_only(
            file_path=file_path, seed=seed, ab_cfg=cfg,
            enable_plot=False, verbose=False
        )

        print("\n===== STATIC COMPARE (scene=0) =====")
        print(f"[MIXED]      cost={res_mixed['cost']:.3f}  truck={res_mixed['truck_dist']:.3f}  drone={res_mixed['drone_dist']:.3f}  time={res_mixed['system_time']:.3f}  late={res_mixed['total_late']:.3f}")
        print(f"[TRUCK-ONLY] cost={res_truck['cost']:.3f}  truck={res_truck['truck_dist']:.3f}  drone={res_truck['drone_dist']:.3f}  time={res_truck['system_time']:.3f}  late={res_truck['total_late']:.3f}")
        return

    # ===== 6) 批量对比：多 seed + 多 cfg（不使用回放）=====
    if RUN_BATCH:
        cfgs = [CFG_A, CFG_D]
        seeds = [2021, 2022, 2023]  # 你也可以先用 [2025] 小跑验证
        enable_plot_batch = False
        verbose_batch = False
        # 公平性说明：
        # - events_path 非空：所有配置共享同一离线请求流（推荐/最公平）
        # - events_path 为空：仅运行场景0（无动态请求）；若要动态对比请提供 events_path
        for sd in seeds:
            for _cfg in cfgs:
                sim.set_simulation_params(road_factor=road_factor)
                res = run_one(
                    file_path=file_path, seed=sd, ab_cfg=_cfg,
                    perturbation_times=perturbation_times,
                    enable_plot=enable_plot_batch, verbose=verbose_batch,
                    events_path=events_path, decision_log_path=''
                )
                r0 = res[0]
                print(f"[BATCH] seed={sd} cfg={_cfg.get('name','?')} cost0={r0['cost']:.3f} time0={r0['system_time']:.3f} late0={r0['total_late']:.3f}")
        return
    
    # ===== 7) 正常动态运行（你平时跑的模式）=====

    results = run_one(
        file_path=file_path, seed=seed, ab_cfg=cfg,
        perturbation_times=perturbation_times,
        enable_plot=enable_plot, verbose=verbose,
        events_path=events_path, decision_log_path=''
    )
    ut.print_summary_table(results)

if __name__ == "__main__":
    main()
