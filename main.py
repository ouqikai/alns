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
import os, time, json, csv, copy
import operators as ops  # 引入算子模块
import simulation as sim  # 引入仿真模块
import utils as ut # 工具模块
import dynamic_logic as dyn  # 动态逻辑
from data_io_1322 import read_data
from viz_utils import visualize_truck_drone, compute_global_xlim_ylim, _normalize_decisions_for_viz

DEBUG_QUICK_FILTER = False

# 中文注释：ALNS 内循环调试（建议仅在定位问题时开启，避免刷屏）
DBG_ALNS = False
# 中文注释：每隔多少次迭代打印一次（dbg_alns=True 时生效）
DBG_EVERY = 50
# ========= 数据集 Schema（节点 + 动态事件）=========
# 说明：nodes.csv 仅包含静态节点字段；动态请求流由 events.csv 单独给出（EVENT_TIME, NODE_ID, NEW_X, NEW_Y, EVENT_CLASS）。
CSV_REQUIRED_COLS = [
    "NODE_ID","NODE_TYPE","ORIG_X","ORIG_Y","DEMAND","READY_TIME","DUE_TIME"
]
CSV_NODE_TYPES = {"central", "base", "customer"}
EPS = 1e-9

DEBUG = False
# 中文注释：迟到定位开关（只建议临时打开，避免刷屏）
DEBUG_LATE = False
DEBUG_LATE_TOPK = 15
DEBUG_LATE_SCENES = None   # 只看 scene=0；想看全部就改成 None
# 中文注释：重插入诊断开关（仅用于定位“某个迟到客户为什么迟到”）
DEBUG_REINS_DIAG = False
# 中文注释：指定某个客户 idx；None 表示自动选择“当前最迟到的卡车客户”
DEBUG_REINS_CID = None

CFG_A = {
    "NAME": "A_paired_baseline",
    "PAIRING_MODE": "paired",
"lambda_late": 50.0,
    "late_hard": 0.8,  # 建议显式写在 cfg 里（要更严就 0.10）
    "late_hard_delta": 1.0,
# ===== 新增：quick_filter 阈值（从 cfg 读取，避免写死不一致）=====
    "qf_cost_max": 30,   # 决策阶段：接受请求的Δcost上限
    "qf_late_max": 0.5,   # 决策阶段：接受请求的Δlate上限（小时）

    # ===== 新增：SA 温度尺度（从 cfg 读取）=====
    "sa_T_start": 50.0,    # SA 初温（要和Δcost量级匹配）
    "sa_T_end": 1.0,       # SA 末温（后期更贪心）

    # ===== 新增：destroy 强度（从 cfg 读取）=====
    "remove_fraction": 0.18,
    "min_remove": 5,
    # 固定两对：等价于你原来“捆绑”的两套组合
    "DESTROYS": ["D_random_route", "D_worst_route", "D_reloc_focus_v2", "D_switch_coverage", "D_late_worst"],
    "REPAIRS": ["R_greedy_only", "R_regret_only", "R_greedy_then_drone", "R_regret_then_drone",
                "R_late_repair_reinsert", "R_base_feasible_drone_first"],
    "ALLOWED_PAIRS": [
        ("D_random_route", "R_greedy_then_drone"),
        ("D_worst_route",  "R_regret_then_drone"),
        ("D_reloc_focus_v2", "R_greedy_then_drone"),
        # ↓↓↓ 核心王牌组合：迟到点直接转无人机 ↓↓↓
        ("D_late_worst",   "R_late_repair_reinsert"),
    ],
"dbg_alns": False,
    "dbg_postcheck": False,
"disable_postcheck": 0,
"lambda_prom": 0.0
}

CFG_D = {
    "NAME": "D_full_structured",
    "PAIRING_MODE": "free",
    "late_hard": 0.1,  # 建议显式写在 cfg 里（要更严就 0.10）
    "late_hard_delta": 1.0,
    # ===== 新增：quick_filter 阈值（从 cfg 读取，避免写死不一致）=====
    "qf_cost_max": 30,   # 决策阶段：接受请求的Δcost上限
    "qf_late_max": 0.5,   # 决策阶段：接受请求的Δlate上限（小时）

    # ===== 新增：SA 温度尺度（从 cfg 读取）=====
    "sa_T_start": 50.0,    # SA 初温（要和Δcost量级匹配）
    "sa_T_end": 1.0,       # SA 末温（后期更贪心）
    "alns_max_iter": 1000,   # 最大跑 1000 代
    "max_no_improve": 1000,   # [新增] 连续 150 代不动就停
    # ===== 新增：destroy 强度（从 cfg 读取）=====
    "remove_fraction": 0.18,
    "min_remove": 5,
    "DESTROYS": ["D_random_route", "D_worst_route", "D_reloc_focus_v2", "D_switch_coverage", "D_late_worst"],
    "REPAIRS": ["R_greedy_only", "R_regret_only", "R_greedy_then_drone", "R_regret_then_drone",
                "R_late_repair_reinsert", "R_base_feasible_drone_first"],
    "dbg_alns": False,
    "dbg_postcheck": False,
"disable_postcheck": 0,
"lambda_prom": 0.0
}

CFG_GA = {
    "NAME": "Baseline_GA",
    "name": "Baseline_GA",     # <--- [补齐] 适配 CSV 输出的 name 字段
    "method": "GA",
    "planner": "GA",
    "ga_max_iter": 150,
    "max_no_improve": 30,
    "ga_pop_size": 50,
    "crossover_rate": 0.8,     # <--- [补齐] 显式记录默认值
    "mutation_rate": 0.2,      # <--- [补齐] 显式记录默认值
    "late_hard": 0.1,
    "qf_cost_max": 30.0,
    "qf_late_max": 0.5,
}
CFG_GUROBI = {
    "name": "GUROBI",
    "planner": "GUROBI",
    "force_truck_mode": False,
    "grb_time_limit": 60,     # 你自己定
    "grb_mip_gap": 0.0,       # 小规模可 0；大规模建议 >0
    "grb_verbose": 0,
    "late_hard": 0.1,
    "qf_cost_max": 30.0,
    "qf_late_max": 0.5,
}

def dprint(*args, **kwargs):
    """统一的调试打印开关，避免到处散落 print"""
    if DEBUG:
        print(*args, **kwargs)

def run_one(file_path: str, seed: int, ab_cfg: dict, perturbation_times=None, enable_plot: bool = False, verbose: bool = True, events_path: str = "", decision_log_path: str = ""):
    if verbose:
        print("[CFG-IN]", ab_cfg.get("PAIRING_MODE"), ab_cfg.get("late_hard"),
          len(ab_cfg.get("DESTROYS", [])), len(ab_cfg.get("REPAIRS", [])))
        print("[CFG-QF]", "qf_cost_max=", ab_cfg.get("qf_cost_max", ab_cfg.get("delta_cost_max", 30.0)),
              "qf_late_max=", ab_cfg.get("qf_late_max", ab_cfg.get("delta_late_max", 0.10)))
        print("[CFG-SA]", "sa_T_start=", ab_cfg.get("sa_T_start", ab_cfg.get("T_start", 50.0)),
              "sa_T_end=", ab_cfg.get("sa_T_end", ab_cfg.get("T_end", 1.0)),
              "remove_fraction=", ab_cfg.get("remove_fraction", 0.10),
              "min_remove=", ab_cfg.get("min_remove", 3))

    if perturbation_times is None:
        perturbation_times = []
    # 统一过滤/去重/排序，避免传入 0 导致重复场景、以及不同运行方式输出不一致
    perturbation_times = ut._normalize_perturbation_times(perturbation_times)
    seed_py_rng, seed_np_rng = "", ""
    if seed is not None:
        ut.set_seed(int(seed))
        try:
            import random, pickle, hashlib
            import numpy as np
        except Exception:
            seed_py_rng, seed_np_rng = "", ""

    ab_cfg = ops.build_ab_cfg(ab_cfg)

    # 中文注释：给 ab_cfg 补齐调试开关默认值（dynamic_logic 会透传给 ALNS）
    try:
        ab_cfg.setdefault("dbg_alns", bool(DBG_ALNS))
        ab_cfg.setdefault("dbg_every", int(DBG_EVERY))
        ab_cfg.setdefault("dbg_planner_sets", False)  # 中文注释：打印 ALNS/GRB 输入集合核对
    except Exception:
        pass
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
    # [FIX] 纯卡车模式拦截：若是 TruckOnly，强制清空无人机客户，全部分给卡车
    if bool(ab_cfg.get("force_truck_mode", False)):
        base_to_drone_customers = {}
        truck_customers = list(getattr(data, "customer_indices", []))
        bases_visit_0 = []  # 也不强制访问基站
        if verbose:
            print("[SCENE 0] Force Truck Mode: Initial solution set to Truck-Only.")
    else:
        base_to_drone_customers, truck_customers = sim.classify_clients_for_drone(data)
        bases_visit_0 = None  # 默认逻辑（None=所有基站）
    if verbose:
        print("需要卡车服务的客户数:", len(truck_customers))
        print("各基站无人机客户数:", {b: len(cs) for b, cs in base_to_drone_customers.items()})
    # ===================== 3) 场景0：跑一次 ALNS（No-RL）=====================
    if verbose:
        print("\n===== Advanced ALNS (No RL, official solution) =====")

    ctx0 = dict(ab_cfg)  # 关键：场景0也吃实验配置（paired/free/算子池）
    ctx0["verbose"] = verbose
    # 中文注释：destroy 下限强度（避免拆了又装回去）；未配置则用默认 3
    ctx0["min_remove"] = int(ab_cfg.get("min_remove", 3))

    # [PROMISE] 场景0不计迟到：避免 late_hard 护栏误伤（即使你实验配置里开启了 late_hard）
    ctx0["late_hard"] = 1e18
    ctx0["late_hard_delta"] = 1e18

    # 提前获取算法类型，用于下面的判断
    method_name = ab_cfg.get("method", "G3")
    is_truck_only = bool(ab_cfg.get("force_truck_mode", False))
    is_promise_file = ut._is_promise_nodes_file(file_path)

    # ===================== 【修改点】收敛曲线：scene0 也输出 converge_*.csv =====================
    try:
        # 这里统一用 ab_cfg.get("trace_converge") 总开关
        if bool(ab_cfg.get("trace_converge", False)):
            trace_dir = str(ab_cfg.get("trace_dir", "outputs"))
            os.makedirs(trace_dir, exist_ok=True)
            ctx0["trace_converge"] = True

            # 【修复】：正确区分 G3 和 TruckOnly 的文件名，防止互相覆盖
            file_method_name = "TruckOnly" if is_truck_only else method_name
            ctx0["trace_csv_path"] = os.path.join(trace_dir, f"converge_{file_method_name}_seed{seed}_scene0.csv")
        else:
            ctx0["trace_converge"] = False
            ctx0["trace_csv_path"] = None
    except Exception:
        ctx0["trace_converge"] = False
        ctx0["trace_csv_path"] = None
    # =========================================================================================

    # 【默认值】：对所有算法（Greedy, TruckOnly, 以及未来你加的任何对比算法），默认不惩罚
    lambda_scene0 = 0.0

    # 【特权判断】：只对“完全体 ALNS”施加约束
    if is_promise_file and (method_name == "G3") and (not is_truck_only):
        lambda_scene0 = 50.0
        if verbose:
            print("    [Scene 0] PROPOSED ALNS (Main): Enforcing promise windows (lambda=50).")
    else:
        if verbose:
            print(
                f"    [Scene 0] Baseline/Other ({method_name}, TruckOnly={is_truck_only}): Ignoring promise (lambda=0).")
    # -----------------------------------------------------------
    t0_start = time.time()
    (best_route,
     best_b2d,
     best_cost,
     best_truck_dist,
     best_drone_dist,
     best_total_late,
     best_truck_time) = dyn.alns_truck_drone(
        data,
        base_to_drone_customers,
        max_iter=int(ab_cfg.get('alns_max_iter', 1000)),
        remove_fraction=float(ab_cfg.get("remove_fraction", 0.10)),
        T_start=float(ab_cfg.get("sa_T_start", ab_cfg.get("T_start", 50.0))),
        T_end=float(ab_cfg.get("sa_T_end", ab_cfg.get("T_end", 1.0))),
        alpha_drone=0.3,
        lambda_late=lambda_scene0,
        truck_customers=truck_customers,
        use_rl=False,
        rl_tau=0.5,
        rl_eta=0.1,
        bases_to_visit=bases_visit_0,
        ctx=ctx0
    )
    t0_end = time.time()
    scene0_runtime = t0_end - t0_start
    if verbose:
        print(f"    [SCENE 0] Runtime={scene0_runtime:.3f} s")
    # ================= [新增补丁：纯卡车路线方向校正] =================
    # 目的：强制翻转“逆序”路线，防止卡车过早服务大ID客户，导致动态请求失效。
    is_truck_only_mode = bool(ab_cfg.get("force_truck_mode", False))

    # 仅对 TruckOnly 且路线包含至少2个客户时生效
    if is_truck_only_mode and len(best_route) > 3:
        # 1. 提取中间客户序列（去除首尾 Depot）
        # best_route 结构通常是 [Depot, C1, C2, ..., Cn, Depot]
        inner_indices = best_route[1:-1]

        # 2. 获取首尾客户的 NODE_ID (通常 ID 越大数据集中时间越晚)
        first_node_id = data.nodes[inner_indices[0]]['node_id']
        last_node_id = data.nodes[inner_indices[-1]]['node_id']

        # 3. 判定：如果是降序 (例如 100 -> ... -> 1)，说明跑反了
        if first_node_id > last_node_id:
            if verbose:
                print(f"[TRUCK-TUNE] 🚨 检测到路线逆序 (ID {first_node_id} -> ... -> {last_node_id})。")
                print(f"[TRUCK-TUNE] 正在翻转路线以匹配事件流，改善请求有效性...")

            # 4. 执行翻转
            # 保持首尾 Depot 不动，中间倒序
            best_route = [best_route[0]] + inner_indices[::-1] + [best_route[-1]]

            # 5. !!! 极其重要：翻转后必须强制重算所有状态 !!!
            # 否则后续动态逻辑会沿用旧的到达时间表，导致逻辑错乱

            # 重新跑一遍系统评估，获取最新状态
            # 注意：这里的 lambda_late 建议用 50.0 以便看清翻转后的真实代价
            _eval_fixed = sim.evaluate_full_system(
                data, best_route, best_b2d,
                alpha_drone=0.3, lambda_late=50.0,
                truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
            )

            # 更新关键变量，供后续动态循环使用
            best_cost = _eval_fixed['cost']
            best_truck_dist = _eval_fixed['truck_dist']
            best_total_late = _eval_fixed['total_late']

            # 这里的 arrival_times 会在下面几行代码被再次计算，
            # 但为了保险起见，这里先更新 best_route 对应的状态。
            if verbose:
                print(f"[TRUCK-TUNE] ✅ 翻转完成。新状态: Cost={best_cost:.3f}, Late={best_total_late:.3f}")
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
    report_lambda = float(ab_cfg.get("lambda_late", 50.0))
    full_eval0 = sim.evaluate_full_system(
        data, best_route, best_b2d,
        alpha_drone=0.3, lambda_late=report_lambda,
        truck_speed=sim.TRUCK_SPEED_UNITS, drone_speed=sim.DRONE_SPEED_UNITS
    )
    # ===================== 【新增：专门为 Greedy 补齐 Scene 0 记录】 =====================
    if bool(ab_cfg.get("trace_converge", False)) and ab_cfg.get("method", "G3") == "G1":
        import csv
        trace_dir = str(ab_cfg.get("trace_dir", "outputs"))
        os.makedirs(trace_dir, exist_ok=True)
        g1_scene0_csv = os.path.join(trace_dir, f"converge_G1_seed{seed}_scene0.csv")
        with open(g1_scene0_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["iter", "best_cost_dist", "best_total_late", "best_truck_dist",
                                              "best_drone_dist"])
            w.writeheader()
            w.writerow({
                "iter": 0,
                "best_cost_dist": float(full_eval0.get("truck_dist_eff", full_eval0["truck_dist"])) + 0.3 * float(
                    full_eval0["drone_dist"]),
                "best_total_late": float(full_eval0["total_late"]),
                "best_truck_dist": float(full_eval0["truck_dist"]),
                "best_drone_dist": float(full_eval0["drone_dist"]),
            })
    # [PROMISE] 场景0：输出 late_prom/late_eff（late_eff 以冻结窗为准）
    _late_dir = (os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
    ut.emit_scene_late_logs(_late_dir, scene_idx=0, decision_time=0.0, data=data, full_route=best_route, full_b2d=best_b2d, full_eval=full_eval0, prefix="", drone_speed=sim.DRONE_SPEED_UNITS)
    # 中文注释：scene=0（初始静态解）也输出迟到分解
    if DEBUG_LATE and ((DEBUG_LATE_SCENES is None) or (0 in DEBUG_LATE_SCENES)):
        # 中文注释：debug_print_lateness_topk 已在 slim 版本中移除（避免控制台大输出拖慢实验）。
        # 如需查看 TopK 迟到客户，请查 late_logs/*.csv（emit_scene_late_logs 会写出）。
        pass
    rec0 = ut._pack_scene_recordrec0 = ut._pack_scene_record(
    0, 0.0, full_eval0,
    num_req=0, num_acc=0, num_rej=0,
    alpha_drone=0.3, lambda_late=report_lambda,
    solver_time=scene0_runtime
)
    rec0.update({"seed_py_rng": seed_py_rng, "seed_np_rng": seed_np_rng})
    scenario_results.append(rec0)

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
        t_prev = 0.0

        # 5.0 动态请求流准备
        reloc_radius = float(ab_cfg.get("reloc_radius", 0.8)) if ab_cfg else 0.8
        if offline_groups is None:
            raise RuntimeError("动态模式需要 events.csv")

        decision_times_list = [float(x) for x in perturbation_times]

        # ========== 核心循环：只需调用 run_decision_epoch ==========
        for decision_time in decision_times_list:
            if decision_time < t_prev - 1e-9:
                raise RuntimeError(f"时间逆序: {t_prev} -> {decision_time}")
            # ====== 可视化需要：保存“决策前”的状态（不要被 data_next 覆盖）======
            data_before_viz = data_cur
            # 决策前仍未完成的无人机客户集合：用于画“原位置黑点”的实心/空心
            drone_set_before_viz = set()
            try:
                for _b, _cs in (full_b2d_cur or {}).items():
                    for _c in _cs:
                        if full_finish_cur.get(_c, float("inf")) > decision_time + 1e-9:
                            drone_set_before_viz.add(int(_c))
            except Exception:
                drone_set_before_viz = set()

            # 调用动态逻辑封装函数
            step_res = dyn.run_decision_epoch(
                decision_time=decision_time,
                t_prev=t_prev,
                scene_idx=scene_idx,
                data_cur=data_cur,
                full_route_cur=full_route_cur,
                full_b2d_cur=full_b2d_cur,
                full_arrival_cur=full_arrival_cur,
                full_depart_cur=full_depart_cur,
                full_finish_cur=full_finish_cur,
                offline_groups=offline_groups,
                nodeid2idx=nodeid2idx,
                ab_cfg=ab_cfg,
                seed=seed,
                verbose=verbose
            )

            # 1. 处理 Early Stop
            if step_res.get('break', False):
                break

            # 3. 收集结果与日志
            step_res['stat_record'].update({"seed_py_rng": seed_py_rng, "seed_np_rng": seed_np_rng})
            scenario_results.append(step_res['stat_record'])

            if 'decision_log_rows' in step_res:
                decision_log_rows.extend(step_res['decision_log_rows'])

            # 4. 输出迟到日志 (主文件负责 I/O 路径)
            _late_dir = (
                os.path.join(os.path.dirname(decision_log_path) or ".", "late_logs") if decision_log_path else "")
            ut.emit_scene_late_logs(
                _late_dir,
                scene_idx=scene_idx,
                decision_time=decision_time,
                data=data_cur,
                full_route=full_route_cur,
                full_b2d=full_b2d_cur,
                full_eval=step_res['full_eval'],
                prefix="",
                drone_speed=sim.DRONE_SPEED_UNITS
            )

            # 5. 可视化 (主文件负责画图)
            if enable_plot and 'viz_pack' in step_res:
                vp = step_res['viz_pack']

                dec_viz = _normalize_decisions_for_viz(data_before_viz, vp['decisions'])

                visualize_truck_drone(
                    vp['data'],
                    vp['route'],
                    vp['b2d'],
                    title=f"Scenario {scene_idx} (t={decision_time:.2f}h)",
                    xlim=global_xlim,
                    ylim=global_ylim,
                    decision_time=decision_time,
                    truck_arrival=step_res['full_arrival_next'],  # ✅ 用 step_res 的 next
                    drone_finish=step_res['full_finish_next'],  # 这里注意用 drone finish map
                    prefix_route=vp['prefix_route'],
                    virtual_pos=vp['virtual_pos'],
                    relocation_decisions=dec_viz,
                    drone_set_before=drone_set_before_viz
                )
            # 2. 更新状态
            data_cur = step_res['data_next']
            full_route_cur = step_res['full_route_next']
            full_b2d_cur = step_res['full_b2d_next']
            full_arrival_cur = step_res['full_arrival_next']
            full_depart_cur = step_res['full_depart_next']
            full_finish_cur = step_res['full_finish_next']
            # 推进
            t_prev = decision_time
            scene_idx += 1

    # ===================== 6) 汇总输出（run_one 里建议不打印，交给 main 或 ablation）=====================

    # ---------- 保存 decision_log（离线 events.csv 模式） ----------
    try:
        if offline_groups is not None and bool(ab_cfg.get("save_decision_log", True)):
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


def run_compare_suite(
        file_path: str,
        seed: int,
        base_cfg: dict,
        perturbation_times=None,
        events_path: str = None,
        out_dir: str = "outputs",
        enable_plot: bool = False,
        verbose: bool = False, target_methods: list = None,
):
    """在同一 nodes/events/seed 下，跑多算法对照，并输出 metrics_timeseries.csv。

    可用方法标签（gname）：
    - TruckOnly：纯卡车模型（force_truck_mode=True，仍用 ALNS 优化）
    - Proposed：你的主方法（混合模型 + ALNS）
    - Greedy：贪心/预规划基线（对应 dynamic_logic 的 G1 分支）
    - GA：遗传算法
    - Gurobi：MILP（planner=GRB）
    通过 target_methods 传入 gname 列表可筛选子集。
    """
    if perturbation_times is None:
        perturbation_times = []

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 固定短文件名：每个 run_dir 里永远叫这个
    compare_csv_path = os.path.join(out_dir, "metrics_timeseries.csv")

    # 统一基础配置（默认关掉大量 DBG 打印）
    cfg_base = dict(base_cfg)
    cfg_base.setdefault("dbg_alns", False)
    cfg_base.setdefault("dbg_postcheck", False)
    cfg_base.setdefault("alns_max_iter", 1000)
    cfg_base.setdefault("save_iter_trace", False)  # 默认不保存单场景迭代轨迹
    cfg_base.setdefault("save_decision_log", False)  # 批量套件默认不保存 decision_log，避免文件爆炸

    # 1. 定义配置
    # -----------------------------------------------
    # G1: Greedy / Preplan (基线：无重排)
    # -----------------------------------------------
    cfg_greedy = dict(cfg_base)
    cfg_greedy.update({
        "name": "Baseline_Greedy",
        "method": "G1", "planner": "GREEDY"
    })

    # -----------------------------------------------
    # TruckOnly: 纯卡车 (消融：无无人机) -> 需要在上一步 dynamic_logic 里加拦截
    # -----------------------------------------------
    cfg_truck = dict(cfg_base)
    cfg_truck.update({
        "name": "ALNS_TruckOnly",
        "planner": "ALNS",          # 明确
        "force_truck_mode": True,
        # 纯卡车可以用较强的算子，保证公平对比
        "alns_max_iter": 1000,
        "DESTROYS": ["D_random_route", "D_worst_route"],
        "REPAIRS": ["R_greedy_only", "R_regret_only"]
    })

    # -----------------------------------------------
    # G3: Proposed (你的主方法)
    # -----------------------------------------------
    cfg_alns_hybrid = dict(cfg_base)
    cfg_alns_hybrid.update({
        "name": "ALNS_Hybrid",
        "planner": "ALNS",
        "force_truck_mode": False,
    })
    # -----------------------------------------------
    # Gurobi: MILP 基线
    # -----------------------------------------------
    cfg_grb = dict(cfg_base)
    cfg_grb.update({
        "name": "Baseline_Gurobi",
        "method": "GRB",
        "planner": "GRB",
    })

    # -----------------------------------------------
    # GA: 遗传算法（继承 base_cfg，避免 lambda_late/qf 阈值等不一致）
    # -----------------------------------------------
    cfg_ga = dict(cfg_base)
    cfg_ga.update(dict(CFG_GA))  # CFG_GA 里有 planner="GA"
    # 2. 构造对比列表（用“对外标签”，与 TARGET_METHODS 保持一致）
    all_groups = [
        ("Proposed", cfg_alns_hybrid),
        ("TruckOnly", cfg_truck),
        ("Greedy", cfg_greedy),
        ("GA", cfg_ga),  # 注意：用 cfg_ga，不要用 CFG_GA
        ("Gurobi", cfg_grb),  # 注意：用 cfg_grb，不要用 CFG_GUROBI
    ]

    if target_methods:
        tm = {str(x).strip().lower() for x in target_methods}
        groups = [g for g in all_groups if str(g[0]).strip().lower() in tm]
    else:
        groups = all_groups

    all_rows = []

    for gname, cfg in groups:
        print(f"\n================= {gname} =================")
        decision_log_path = os.path.join(out_dir, f"decision_log_{gname}_{base_name}_seed{seed}.csv")
        res = run_one(
            file_path=file_path,
            seed=seed,
            ab_cfg=cfg,
            perturbation_times=perturbation_times,
            enable_plot=enable_plot,
            verbose=verbose,
            events_path=events_path,
            decision_log_path=decision_log_path
        )
        ut.print_summary_table(res)

        # [修改 1] 将 run_one 返回的列表重命名为 history，避免与下面的单步结果混淆
        history = res
        # [修改 2] 循环变量命名为 step_res (单步结果)
        for t, step_res in enumerate(history):
            if step_res is None: continue

            # [Risk 1 修复] 强制统一 Cost 口径 (Base Cost + Penalty)
            # 这里的 step_res 是字典，不会再报 'list' object has no attribute 'get'
            truck_d = float(step_res.get('truck_dist', 0.0))
            drone_d = float(step_res.get('drone_dist', 0.0))
            total_l = float(step_res.get('total_late', 0.0))

            # 读取参数 (优先从 cfg 读取，保证和决策时一致)
            # 假设 alpha 固定为 0.3
            alpha_val = 0.3
            lam_val = float(cfg.get('lambda_late', 50.0))

            # 重算标准指标
            base_cost_calc = truck_d + alpha_val * drone_d
            penalty_calc = lam_val * total_l
            obj_cost_calc = base_cost_calc + penalty_calc

            # 构建写入行
            row = dict(step_res)  # 复制原始数据
            row.update({
                "method": gname, "algo_seed": int(seed), "dataset_id": base_name,
                "seed": int(seed), "time_step": row.get("time_step", row.get("scene", row.get("scene_idx", ""))),
                # 新增：模型变体（默认 hybrid；纯卡车时你在 cfg 里设成 truck_only）
                # 模型变体：由 force_truck_mode 推断，避免你漏配
                "model_variant": ("truck_only" if bool(cfg.get("force_truck_mode", False)) else "hybrid"),

                # 算法名：就用本轮组名（gname），最干净
                "algo_name": gname,
                # 增强名字提取鲁棒性 (优先取 name, 其次 NAME)
                "cfg_name": str(cfg.get("name", cfg.get("NAME", ""))),

                # [Risk 1] 核心修正：覆盖为统一口径
                "base_cost": base_cost_calc,
                "penalty": penalty_calc,
                "cost": obj_cost_calc,  # <--- 修正后的总目标函数值

                # 补充配置参数列 (ALNS/GA 通用)
                "qf_cost_max": cfg.get("qf_cost_max", cfg.get("delta_cost_max", "")),
                "qf_late_max": cfg.get("qf_late_max", cfg.get("delta_late_max", "")),
                "remove_fraction": cfg.get("remove_fraction", ""),
                "min_remove": cfg.get("min_remove", ""),
                "sa_T_start": cfg.get("sa_T_start", ""),
                "sa_T_end": cfg.get("sa_T_end", ""),
                "late_hard": cfg.get("late_hard", ""),  # [Risk 3] 记录硬约束阈值
                "alns_max_iter": cfg.get("alns_max_iter", ""),

                # GA 专属参数
                "ga_pop_size": cfg.get("ga_pop_size", ""),
                "ga_max_iter": cfg.get("ga_max_iter", ""),
                "ga_cx_rate": cfg.get("crossover_rate", ""),
                "ga_mut_rate": cfg.get("mutation_rate", ""),
            })
            all_rows.append(row)

    # 写 CSV（字段取并集，保证不丢信息）
    fields = []
    seen = set()
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)

    with open(compare_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"[COMPARE] written: {compare_csv_path} (rows={len(all_rows)})")
    return compare_csv_path

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
    # file_path = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\nodes_25_seed2023_20260129_164341_promise.csv"
    # events_path = r"D:\代码\ALNS+DL\exp\datasets\25_data\2023\events_25_seed2023_20260129_164341.csv"
    # file_path = r"D:\代码\ALNS+DL\exp\datasets\50_data\2023\nodes_50_seed2023_20260129_174717_promise.csv"
    # events_path = r"D:\代码\ALNS+DL\exp\datasets\50_data\2023\events_50_seed2023_20260129_174717.csv"
    # file_path = r"D:\代码\ALNS+DL\exp\datasets\100_data\2023\nodes_100_seed2023_20260129_190818_promise.csv"
    # events_path = r"D:\代码\ALNS+DL\exp\datasets\100_data\2023\events_100_seed2023_20260129_190818.csv"
    file_path = r"D:\代码\ALNS+DL\nodes_200_seed2023_20260203_142507.csv"
    events_path = r"D:\代码\ALNS+DL\events_200_seed2023_20260203_142507.csv"
    seed = 2025
    cfg = dict(CFG_D)
    cfg.update({"use_rl": False,          # <--- 开启 RL
        "rl_eta": 0.1,
        "reloc_focus_mode": "rej_first",
        "drone_first_pick": "min_obj",
    })

    # cfg["planner"] = "GRB"  # 让 dynamic_logic 走 gurobi 分支
    cfg["planner"] = "ALNS"
    cfg["grb_time_limit"] = 1800  # 每个决策点的 MILP 限时（秒）
    cfg["grb_mip_gap"] = 0.00  # 可选
    cfg["grb_verbose"] = 0  # 可选：0 安静，1 输出更多
    cfg["trace_converge"] = bool(cfg.get("save_iter_trace", False))
    if cfg["trace_converge"]:
        cfg["trace_csv_path"] = "outputs"
    else:
        cfg.pop("trace_csv_path", None)

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
    print(f"[PARAM] TRUCK_ROAD_FACTOR={sim.TRUCK_ROAD_FACTOR:.3f}; TRUCK_SPEED_UNITS={TRUCK_SPEED_UNITS:.3f} units/h (fixed); truck_arc = euclid * {sim.TRUCK_ROAD_FACTOR:.3f}")

    # ===== 3) 运行模式开关 =====
    # 3.4 对照组套件：G0–G3（动态对比）
    RUN_COMPARE_SUITE = False

    if RUN_COMPARE_SUITE:
        import os, time, csv, copy

        # 你要跑的 5 个 algo_seed
        SEEDS_TO_RUN = [2021, 2022, 2023, 2024, 2025]

        # 选择对比层面：模型(model) 或 算法(algo)
        SUITE_LEVEL = "model"  # "model" 或 "algo"

        if SUITE_LEVEL == "model":
            # 模型层面：纯卡车 vs 混合（主方法）
            TARGET_METHODS = ["TruckOnly", "Proposed"]
        elif SUITE_LEVEL == "algo":
            # 算法层面：固定混合模型，对比多算法
            TARGET_METHODS = ["Gurobi", "Greedy", "GA", "Proposed"]
        else:
            TARGET_METHODS = None  # 全跑（一般不建议）

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        suite_dir = os.path.join("outputs", "suites", f"suite_{SUITE_LEVEL}_{base_name}_{ts}")
        os.makedirs(suite_dir, exist_ok=True)

        index_rows = []
        for algo_seed in SEEDS_TO_RUN:
            run_dir = os.path.join(suite_dir, f"seed_{algo_seed}")
            os.makedirs(run_dir, exist_ok=True)

            # 深拷贝，避免 cfg 串组污染
            cfg_run = copy.deepcopy(cfg)
            print("[SUITE] TARGET_METHODS =", TARGET_METHODS)
            csv_path = run_compare_suite(
                file_path=file_path,
                seed=algo_seed,
                base_cfg=cfg_run,
                perturbation_times=perturbation_times,
                events_path=events_path,
                out_dir=run_dir,
                enable_plot=False,
                verbose=False,
                target_methods=TARGET_METHODS,
            )
            index_rows.append({"algo_seed": algo_seed, "csv": os.path.relpath(csv_path, suite_dir)})

        # 写总索引：你后续画图只需要读这个
        index_path = os.path.join(suite_dir, "suite_index.csv")
        with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["algo_seed", "csv"])
            w.writeheader()
            w.writerows(index_rows)

        print("[SUITE] suite_dir =", suite_dir)
        print("[SUITE] index =", index_path)
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
