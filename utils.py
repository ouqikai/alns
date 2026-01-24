# utils.py
import os
import csv
import random
import numpy as np
import math
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# =========================
# 冻结承诺时间窗（Promise Window）相关工具
# =========================
def _derive_promise_nodes_path(in_nodes_csv: str) -> str:
    """中文注释：nodes_xxx.csv -> nodes_xxx_promise.csv（不覆盖原文件）。若已是 *_promise.csv 则原样返回。"""
    p = str(in_nodes_csv)
    pl = p.lower()
    if pl.endswith("_promise.csv"):
        return p
    if pl.endswith(".csv"):
        return p[:-4] + "_promise.csv"
    return p + "_promise.csv"

def _is_promise_nodes_file(nodes_path: str) -> bool:
    """中文注释：判断输入 nodes 文件是否已经是冻结承诺窗版本（*_promise.csv）。"""
    try:
        return str(nodes_path).lower().endswith("_promise.csv")
    except Exception:
        return False

def freeze_existing_promise_windows_inplace(data):
    """中文注释：
    若输入本身就是 *_promise.csv，则认为 READY_TIME/DUE_TIME 已是冻结的“平台承诺窗”。

    目标：只保留两列时间（READY_TIME/DUE_TIME）作为承诺窗；运行态通过 effective_due 表示“有效截止”。
    - promised ready  = node['ready_time']
    - promised due    = node['due_time']
    - effective due   = node.get('effective_due', node['due_time'])

    这里做字段初始化：为所有 customer 补齐 effective_due（默认=承诺 due）。
    """
    for n in getattr(data, "nodes", []):
        if str(n.get("node_type", "")).lower() != "customer":
            continue
        try:
            pd = float(n.get("due_time", 0.0))
        except Exception:
            pd = 0.0
        # 中文注释：未发生成功变更时，有效截止=承诺截止
        n["effective_due"] = float(n.get("effective_due", pd))

# =========================
# 时间窗统一读写入口（收口）
# =========================
def _to_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def prom_ready(node: dict) -> float:
    """中文注释：平台承诺 READY（冻结窗）"""
    if node is None:
        return 0.0
    return _to_float(node.get("ready_time", 0.0), 0.0)

def prom_due(node: dict) -> float:
    """中文注释：平台承诺 DUE（冻结窗）"""
    if node is None:
        return float("inf")
    # 兜底：若 due_time 缺失，就至少不小于 ready
    pr = prom_ready(node)
    pd = _to_float(node.get("due_time", pr), pr)
    return max(pd, pr)

def eff_due(node: dict) -> float:
    """中文注释：运行态有效截止（优先 effective_due，否则回退 prom_due）"""
    if node is None:
        return float("inf")
    v = node.get("effective_due", None)
    if v is None:
        return prom_due(node)
    return _to_float(v, prom_due(node))

def cand_eff_due(node: dict) -> float:
    """中文注释：候选有效截止 L = max(prom_due, prom_ready + delta_avail_h)"""
    if node is None:
        return float("inf")
    pr = prom_ready(node)
    pd = prom_due(node)
    dav = _to_float(node.get("delta_avail_h", 0.0), 0.0)
    return max(pd, pr + dav)

def apply_promise_windows_inplace(data, eta_map: dict, promise_width_h: float = 0.5):
    """中文注释：
    用场景0解得到的 ETA 冻结承诺窗，并写回 data.nodes[*].ready_time / due_time。
    兼容两种 eta_map 口径：
      - key 是节点下标 idx（最常见）
      - key 是 NODE_ID（瘦身/改写时容易变成这种）
    """
    for i, n in enumerate(data.nodes):
        if str(n.get("node_type", "")).lower() != "customer":
            continue

        # 1) 优先按 idx 取 ETA；取不到再按 NODE_ID 取
        if i in eta_map:
            pr = float(eta_map[i])
        else:
            try:
                nid = int(n.get("node_id", -1))
            except Exception:
                nid = -1
            if nid in eta_map:
                pr = float(eta_map[nid])
            else:
                # 2) 兜底：仍取原 ready_time（或 0）
                pr = float(n.get("ready_time", 0.0))

        pd = pr + float(promise_width_h)

        # 内存里保留这些字段，便于 decision_log / debug（不要求写回 CSV）
        n["prom_ready"] = pr
        n["prom_due"] = pd
        n["effective_due"] = pd

        # 关键：冻结窗写回两列时间（你要的“只留两列时间”）
        n["ready_time"] = pr
        n["due_time"] = pd

def write_promise_nodes_csv(in_nodes_csv: str, out_nodes_csv: str, eta_map: dict, promise_width_h: float = 0.5):
    """中文注释：
    输出 nodes_*_promise.csv（可选），但仍只保留 READY_TIME / DUE_TIME 两列时间，不新增 PROM_* 列，
    避免 strict_schema=True 时读入失败。
    同时兼容 eta_map 的 idx / NODE_ID 两种 key。
    """
    if (out_nodes_csv is None) or (str(out_nodes_csv).strip() == ""):
        return
    os.makedirs(os.path.dirname(out_nodes_csv) or ".", exist_ok=True)

    with open(in_nodes_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = list(r)

    with open(out_nodes_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row_idx, row in enumerate(rows):
            ntype = str(row.get("NODE_TYPE", "")).strip().lower()
            if ntype != "customer":
                w.writerow(row)
                continue

            try:
                nid = int(float(row.get("NODE_ID", -1)))
            except Exception:
                nid = -1

            pr = None
            if row_idx in eta_map:
                pr = float(eta_map[row_idx])
            elif nid in eta_map:
                pr = float(eta_map[nid])

            if pr is not None:
                pd = pr + float(promise_width_h)
                row["READY_TIME"] = f"{pr:.3f}"
                row["DUE_TIME"] = f"{pd:.3f}"

            w.writerow(row)

def load_events_csv(path: str):
    """读取 events.csv（仅事实，不含 accept/reject）。"""
    events = []
    if (path is None) or (str(path).strip() == ""):
        return events
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def _f(k, default=0.0):
                try:
                    return float(row.get(k, default))
                except Exception:
                    return float(default)
            def _i(k, default=0):
                try:
                    return int(float(row.get(k, default)))
                except Exception:
                    return int(default)

            ev = {
                "EVENT_ID": _i("EVENT_ID", 0),
                "EVENT_TIME": _f("EVENT_TIME", 0.0),
                "NODE_ID": _i("NODE_ID", -1),
                "NEW_X": _f("NEW_X", 0.0),
                "NEW_Y": _f("NEW_Y", 0.0),
                "EVENT_CLASS": str(row.get("EVENT_CLASS", "")).strip().upper(),
                "DELTA_AVAIL_H": _f("DELTA_AVAIL_H", 0.0),
            }
            events.append(ev)
    return events

def group_events_by_time(events, ndigits: int = 6):
    """按 EVENT_TIME 分组（key 使用 round(t, ndigits)）。"""
    g = {}
    for e in events:
        t = round(float(e.get("EVENT_TIME", 0.0)), ndigits)
        g.setdefault(t, []).append(e)
    return g

def map_event_class_to_reloc_type(event_class: str) -> str:
    """将 events.csv 的 EVENT_CLASS 映射到求解端内部 reloc_type."""
    s = str(event_class or "").strip().upper()
    if s in ("IN_DB", "INTRA", "INTRA_DB"):
        return "intra"
    if s in ("CROSS_DB", "CROSS", "CROSSBASE", "CROSS_BASE"):
        return "cross"
    if s in ("OUT_DB", "OUT", "OUTSIDE"):
        return "out"
    # 默认：legacy（沿用旧逻辑）
    return "legacy"

def save_decision_log(rows, path: str):
    """保存 decision_log.csv：事件决策日志（含承诺窗/有效窗与Δ指标）。"""
    if (path is None) or (str(path).strip() == ""):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "EVENT_ID", "EVENT_TIME", "NODE_ID",
        "DECISION", "REASON",
        "OLD_X", "OLD_Y", "NEW_X", "NEW_Y",
        "EVENT_CLASS",
        "APPLIED_X", "APPLIED_Y",
        "FORCE_TRUCK", "BASE_LOCK",
        "DELTA_AVAIL_H",
        "READY_TIME", "DUE_TIME",
        "EFFECTIVE_DUE",
        "D_COST", "D_LATE_PROM", "D_LATE_EFF"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            # 兼容缺失字段
            out = {k: r.get(k, "") for k in fields}
            w.writerow(out)

def print_tw_stats(data):
    """
    中文注释：打印时间窗统计；若识别不到 customer，会输出示例节点帮助定位字段名/取值
    """
    nodes = getattr(data, "nodes", None)
    if not nodes:
        print("[TW] data.nodes 为空或不存在")
        return

    # 先必打印：节点数量与一个样例节点的 key
    sample = nodes[0] if len(nodes) > 0 else {}
    # print(f"[TW] nodes={len(nodes)} sample_keys={list(sample.keys())[:12]} sample={ {k: sample.get(k) for k in list(sample.keys())[:6]} }")

    def get_field(n, *keys, default=None):
        for k in keys:
            if k in n and n[k] is not None:
                return n[k]
        return default

    ready, due, w = [], [], []
    cust_cnt = 0

    for n in nodes:
        # 兼容不同字段名
        nt = get_field(n, "NODE_TYPE", "node_type", "TYPE", "type", default="")
        nt = str(nt).strip().lower()

        # 兼容 customer 的多种写法
        is_customer = (nt == "customer") or (nt == "c") or ("cust" in nt)

        if not is_customer:
            continue

        cust_cnt += 1
        r = float(get_field(n, "READY_TIME", "ready_time", "READY", "ready", default=0.0))
        d = float(get_field(n, "DUE_TIME", "due_time", "DUE", "due", default=0.0))
        ready.append(r)
        due.append(d)
        w.append(max(0.0, d - r))

    if cust_cnt == 0:
        # 再给一个 customer 候选排查：看前几个节点的类型字段值
        types_preview = []
        for i in range(min(5, len(nodes))):
            types_preview.append(get_field(nodes[i], "NODE_TYPE", "node_type", "TYPE", "type", default=None))
        print(f"[TW] 未识别到 customer 节点。前5个节点的 type 字段预览={types_preview}")
        return

    print(f"[TW] customers={cust_cnt} "
          f"ready(min/mean/max)={min(ready):.2f}/{np.mean(ready):.2f}/{max(ready):.2f}  "
          f"due(min/mean/max)={min(due):.2f}/{np.mean(due):.2f}/{max(due):.2f}  "
          f"width(mean)={np.mean(w):.2f}h")

# utils.py (追加在末尾)

def compute_eta_map(data, full_route, full_b2d, full_eval, *, drone_speed=None):
    """中文注释：计算每个客户的 ETA（truck=到达/服务时刻；drone=单程到达客户时刻）。
    注意：在 utils 中不依赖全局变量，请务必传入 drone_speed。
    """
    if drone_speed is None:
        # 兜底：防止调用方忘记传，但这在 utils 里是不安全的，建议调用方必传
        # 这里为了兼容性，若未传则抛出明确错误提示
        raise ValueError("[utils] compute_eta_map 必须传入 drone_speed 参数")

    eta = {}
    # truck 客户
    arr = full_eval.get("arrival", {}) or {}
    for idx in full_route:
        if 0 <= int(idx) < len(data.nodes) and str(data.nodes[int(idx)].get("node_type","")).lower() == "customer":
            eta[int(idx)] = float(arr.get(int(idx), 0.0))
    # drone 客户
    dep = full_eval.get("depart", {}) or {}
    for b, cs in (full_b2d or {}).items():
        for c in cs:
            try:
                c = int(c); b = int(b)
            except Exception:
                continue
            if not (0 <= c < len(data.nodes) and 0 <= b < len(data.nodes)):
                continue
            d_bc = float(data.costMatrix[b, c])
            eta[c] = float(dep.get(c, 0.0)) + d_bc / float(drone_speed)
    return eta

def _total_late_against_due(data, full_route, full_b2d, full_eval, *, due_mode: str = "prom", drone_speed=None):
    """中文注释：计算 total_late（truck+drone），用于 promised lateness 与 effective lateness 对比。

    约定：
    - due_mode='prom' -> 使用节点 due_time（平台承诺截止）
    - due_mode='eff'  -> 使用节点 effective_due（若缺失则回退 due_time）
    """
    if drone_speed is None:
        raise ValueError("[utils] _total_late_against_due 必须传入 drone_speed 参数")

    arr = full_eval.get("arrival", {}) or {}
    dep = full_eval.get("depart", {}) or {}
    total = 0.0

    def _get_due(c):
        n = data.nodes[c]
        if due_mode == "eff":
            try:
                return float(n.get("effective_due", n.get("due_time", float("inf"))))
            except Exception:
                return float("inf")
        try:
            return float(n.get("due_time", float("inf")))
        except Exception:
            return float("inf")

    # truck
    for idx in full_route:
        if 0 <= int(idx) < len(data.nodes) and str(data.nodes[int(idx)].get("node_type","")).lower() == "customer":
            c = int(idx)
            eta = float(arr.get(c, 0.0))
            due = _get_due(c)
            if eta > due:
                total += (eta - due)
    # drone
    for b, cs in (full_b2d or {}).items():
        for c in cs:
            try:
                c = int(c); b = int(b)
            except Exception:
                continue
            if not (0 <= c < len(data.nodes) and 0 <= b < len(data.nodes)):
                continue
            eta = float(dep.get(c, 0.0)) + float(data.costMatrix[b, c]) / float(drone_speed)
            due = _get_due(c)
            if eta > due:
                total += (eta - due)
    return float(total)

def emit_scene_late_logs(out_dir: str, scene_idx: int, decision_time: float, data, full_route, full_b2d, full_eval, *, drone_speed=None, prefix: str = ""):
    """中文注释：每个场景输出 late_prom / late_eff 汇总与明细 CSV（便于后续确定硬拒绝/软惩罚策略）。"""
    if drone_speed is None:
        raise ValueError("[utils] emit_scene_late_logs 必须传入 drone_speed 参数")

    late_prom = _total_late_against_due(data, full_route, full_b2d, full_eval, due_mode="prom", drone_speed=drone_speed)
    late_eff = _total_late_against_due(data, full_route, full_b2d, full_eval, due_mode="eff", drone_speed=drone_speed)

    # 汇总打印
    n_cust = sum(1 for n in data.nodes if str(n.get("node_type","")).lower() == "customer")
    print(f"[LATE-SCENE] scene={scene_idx} t={decision_time:.2f}h customers={n_cust} late_prom={late_prom:.3f} late_eff={late_eff:.3f}")

    if (out_dir is None) or (str(out_dir).strip() == ""):
        return {"late_prom": late_prom, "late_eff": late_eff}

    os.makedirs(out_dir, exist_ok=True)
    fn = f"{prefix}late_scene{scene_idx:02d}_t{decision_time:.2f}.csv".replace(":", "_")
    out_csv = os.path.join(out_dir, fn)

    arr = full_eval.get("arrival", {}) or {}
    dep = full_eval.get("depart", {}) or {}
    # build rows
    rows = []
    # truck customers
    truck_set = set([int(i) for i in full_route if 0 <= int(i) < len(data.nodes) and str(data.nodes[int(i)].get("node_type","")).lower()=="customer"])
    drone_set = set([int(c) for cs in (full_b2d or {}).values() for c in cs])
    for i, n in enumerate(data.nodes):
        if str(n.get("node_type","")).lower() != "customer":
            continue
        mode = "truck" if i in truck_set else ("drone" if i in drone_set else "uncovered")
        if mode == "truck":
            eta = float(arr.get(i, 0.0))
        elif mode == "drone":
            # find base
            b_found = None
            for b, cs in (full_b2d or {}).items():
                if i in cs:
                    b_found = int(b); break
            if b_found is None:
                eta = float(dep.get(i, 0.0))
            else:
                eta = float(dep.get(i, 0.0)) + float(data.costMatrix[b_found, i]) / float(drone_speed)
        else:
            eta = float("nan")
        prom_due = float(n.get("due_time", float("nan")))
        eff_due = float(n.get("effective_due", prom_due))
        late_p = 0.0 if (math.isnan(eta) or math.isnan(prom_due)) else max(0.0, eta - prom_due)
        late_e = 0.0 if (math.isnan(eta) or math.isnan(eff_due)) else max(0.0, eta - eff_due)
        rows.append({
            "IDX": i,
            "NODE_ID": int(n.get("node_id", i)),
            "MODE": mode,
            "ETA_T": eta,
            "PROM_DUE": prom_due,
            "EFFECTIVE_DUE": eff_due,
            "LATE_PROM": late_p,
            "LATE_EFF": late_e,
        })

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return {"late_prom": late_prom, "late_eff": late_eff, "csv": out_csv}

def _pack_scene_record(scene_idx, t_dec, full_eval, num_req, num_acc, num_rej,
                       alpha_drone=0.3, lambda_late=50.0, solver_time=0.0):
    """统一封装动态场景记录行，确保关键指标口径一致。"""
    base_cost = float(full_eval.get("truck_dist_eff", full_eval["truck_dist"]))                 + float(alpha_drone) * float(full_eval["drone_dist"])
    penalty = float(lambda_late) * float(full_eval["total_late"])
    return {
        "scene": int(scene_idx),
        "t_dec": float(t_dec),
        "cost": float(full_eval["cost"]),
        "base_cost": base_cost,
        "penalty": penalty,
        "lambda_late": float(lambda_late),
        "truck_dist": float(full_eval["truck_dist"]),
        "drone_dist": float(full_eval["drone_dist"]),
        "system_time": float(full_eval["system_time"]),
        "truck_late": float(full_eval["truck_late"]),
        "drone_late": float(full_eval["drone_late"]),
        "total_late": float(full_eval["total_late"]),
        "num_req": int(num_req),
        "num_acc": int(num_acc),
        "num_rej": int(num_rej),
        "solver_time": float(solver_time)
    }

# ===================== 纯卡车 vs 卡车-无人机（静态距离/成本对比）=====================
def print_summary_table(scenario_results):
    print("\n===== 动态位置变更场景汇总 =====")
    print("scene | t_dec | cost(obj) | base_cost | penalty | truck_dist | drone_dist | system_time | total_late | req | acc | rej | runtime")
    for rec in scenario_results:
        base_cost = rec.get("base_cost", None)
        penalty = rec.get("penalty", None)

        # 兼容旧结果：若未提供拆分，则用 cost 与 total_late 反推（前提：lambda_late 在 rec 或使用默认 50.0）
        if base_cost is None or penalty is None:
            lam = float(rec.get("lambda_late", 50.0))
            try:
                penalty = lam * float(rec["total_late"])
                base_cost = float(rec["cost"]) - penalty
            except Exception:
                penalty = 0.0
                base_cost = float(rec.get("cost", 0.0))
        run_time = rec.get("solver_time", 0.0)
        print(f"{rec['scene']:5d} | "
              f"{rec['t_dec']:5.2f} | "
              f"{rec['cost']:8.3f} | "
              f"{base_cost:8.3f} | "
              f"{penalty:7.3f} | "
              f"{rec['truck_dist']:10.3f} | "
              f"{rec['drone_dist']:10.3f} | "
              f"{rec['system_time']:11.3f} | "
              f"{rec['total_late']:9.3f} | "
              f"{rec['num_req']:3d} | "
              f"{rec['num_acc']:3d} | "
              f"{rec['num_rej']:3d} | "
              f"{run_time:11.3f}")

def _print_operator_stats(op_stat: dict, top_k: int = 20):
    """打印 ALNS 算子统计（calls/accepts/sa_reject/late_fail 等），用于验证消融是否真起作用。"""
    if not op_stat:
        print("[ALNS-SUM] op_stat empty")
        return
    rows = []
    for k, s in op_stat.items():
        calls = int(s.get("calls", 0))
        acc = int(s.get("accepts", 0))
        sarej = int(s.get("sa_reject", 0))
        latef = int(s.get("late_fail", 0))
        latedf = int(s.get("late_delta_fail", 0))
        repf = int(s.get("repair_fail", 0))
        covf = int(s.get("cover_fail", 0))
        besth = int(s.get("best_hits", 0))
        bestg = float(s.get("best_gain", 0.0))
        acc_rate = (acc / calls) if calls > 0 else 0.0
        best_rate = (besth / calls) if calls > 0 else 0.0
        rows.append((k, calls, acc, acc_rate, besth, best_rate, sarej, latef, latedf, repf, covf, bestg))
    rows.sort(key=lambda x: (-x[3], -x[5], -x[11]))
    print("[ALNS-SUM] op_key | calls acc acc_rate best_hits best_rate sa_rej late_abs_fail late_delta_fail repair_fail cover_fail best_gain")
    for r in rows[:top_k]:
        print("  - %s | %4d %4d %.2f %4d %.2f %4d %4d %4d %4d %4d %.3f" %
              (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]))

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

