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
    若输入本身就是 *_promise.csv，则认为 READY_TIME/DUE_TIME 已是冻结的承诺窗（PROM_READY/PROM_DUE）。
    这里仅做字段初始化，确保后续事件处理与日志统计能读到 prom_ready/prom_due/effective_due。
    """
    for n in getattr(data, "nodes", []):
        if str(n.get("node_type", "")).lower() != "customer":
            continue
        try:
            pr = float(n.get("ready_time", 0.0))
        except Exception:
            pr = 0.0
        try:
            pd = float(n.get("due_time", 0.0))
        except Exception:
            pd = pr
        n["prom_ready"] = pr
        n["prom_due"] = pd
        n["effective_due"] = pd
        # 中文注释：求解端后续一律用当前 due_time 作为“有效截止时间”
        n["due_time"] = pd

def apply_promise_windows_inplace(data, eta_map: dict, promise_width_h: float = 0.5):
    """中文注释：将 PROM_READY/PROM_DUE 写入节点，并冻结为后续场景默认时间窗；同时把 due_time 置为有效截止时间。"""
    for i, n in enumerate(data.nodes):
        if str(n.get("node_type","")).lower() != "customer":
            continue
        if i not in eta_map:
            # 中文注释：极端情况下可能未被分配；兜底用现有 ready/due
            pr = float(n.get("ready_time", 0.0))
        else:
            pr = float(eta_map[i])
        pd = pr + float(promise_width_h)
        n["prom_ready"] = pr
        n["prom_due"] = pd
        n["effective_due"] = pd
        # 中文注释：为了让现有 evaluate_full_system/ALNS 直接使用“冻结窗”，此处把 ready/due 写回 ready_time/due_time
        n["ready_time"] = pr
        n["due_time"] = pd

def write_promise_nodes_csv(in_nodes_csv: str, out_nodes_csv: str, eta_map: dict, promise_width_h: float = 0.5):
    """中文注释：输出 nodes_*_promise.csv，不覆盖原始数据集。"""
    if (out_nodes_csv is None) or (str(out_nodes_csv).strip() == ""):
        return
    os.makedirs(os.path.dirname(out_nodes_csv) or ".", exist_ok=True)
    with open(in_nodes_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        # 确保 PROM_* 列存在
        if "PROM_READY" not in fieldnames:
            fieldnames.append("PROM_READY")
        if "PROM_DUE" not in fieldnames:
            fieldnames.append("PROM_DUE")
        rows = []
        for row in r:
            rows.append(row)
    with open(out_nodes_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            ntype = str(row.get("NODE_TYPE","")).strip().lower()
            try:
                nid = int(float(row.get("NODE_ID", -1)))
            except Exception:
                nid = -1
            if ntype == "customer" and nid in eta_map:
                pr = float(eta_map[nid])
                pd = pr + float(promise_width_h)
                row["READY_TIME"] = f"{pr:.3f}"
                row["DUE_TIME"] = f"{pd:.3f}"
                row["PROM_READY"] = f"{pr:.3f}"
                row["PROM_DUE"] = f"{pd:.3f}"
            w.writerow(row)

# =========================
# 在线扰动日志（可复现）
# =========================
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
        "PROM_READY", "PROM_DUE",
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
    """中文注释：计算 total_late（truck+drone），用于 late_prom 与 late_eff 对比。
    due_mode='prom' -> 使用节点 prom_due（若缺失回退 due_time）
    due_mode='eff'  -> 使用节点 due_time（即当前有效截止时间）
    """
    if drone_speed is None:
        raise ValueError("[utils] _total_late_against_due 必须传入 drone_speed 参数")

    arr = full_eval.get("arrival", {}) or {}
    dep = full_eval.get("depart", {}) or {}
    total = 0.0

    def _get_due(c):
        n = data.nodes[c]
        if due_mode == "prom":
            return float(n.get("prom_due", n.get("due_time", float("inf"))))
        return float(n.get("due_time", float("inf")))

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
        prom_due = float(n.get("prom_due", float("nan")))
        eff_due = float(n.get("due_time", float("nan")))
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