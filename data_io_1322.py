
# data_io_1529.py
# 说明：
# - 仅负责“数据读入 + 基础一致性校验 + 生成 costMatrix + 最近基站(base_id)”
# - 不再依赖 GRB_dec / 主文件里的全局变量（如 Q_drone、UAV_endurance）
# - 你最终确定的数据集列：
#   NODE_ID,NODE_TYPE,ORIG_X,ORIG_Y,PERTURBED_X,PERTURBED_Y,DEMAND,READY_TIME,DUE_TIME,REQUEST_TIME

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ====== 1) 固定数据集 Schema（护栏 1：列锁定）======
# ====== 1) 固定数据集 Schema（护栏 1：列锁定）======
# 在线扰动模式说明：
# - 数据集阶段不再生成 PERTURBED_X/PERTURBED_Y/REQUEST_TIME；
# - 这三列改为“可选列”，若不存在则在 read_data 中自动补齐为 NaN。
CSV_REQUIRED_COLS: List[str] = [
    "NODE_ID",
    "NODE_TYPE",      # central / base / customer
    "ORIG_X",
    "ORIG_Y",
    "DEMAND",
    "READY_TIME",
    "DUE_TIME",
]

CSV_OPTIONAL_COLS: List[str] = [
    "PERTURBED_X",
    "PERTURBED_Y",
    "REQUEST_TIME",   # 可空
]


ALLOWED_NODE_TYPES = {"central", "base", "customer"}


def safe_float(x) -> float:
    """将可能为空的值转为 float；空/NaN/空字符串 -> nan。"""
    if x is None:
        return float("nan")
    if isinstance(x, str) and x.strip() == "":
        return float("nan")
    try:
        if pd.isna(x):
            return float("nan")
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass
class Data:
    nodes: List[dict] = field(default_factory=list)
    node_id_to_idx: Dict[int, int] = field(default_factory=dict)
    costMatrix: Optional[np.ndarray] = None
    central_idx: Optional[int] = None
    base_indices: List[int] = field(default_factory=list)
    customer_indices: List[int] = field(default_factory=list)
    UAVCustomerinDCRange: List[int] = field(default_factory=list)
    schema_cols: List[str] = field(default_factory=list)


def _validate_input_df(df: pd.DataFrame, strict_schema: bool = True) -> None:
    """护栏 2：输入 CSV 的列/值/类型做强校验。"""
    cols = list(df.columns)

    missing = [c for c in CSV_REQUIRED_COLS if c not in cols]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}；当前列: {cols}")

    # 可选列缺失则自动补齐为 NaN（在线扰动模式默认不提供这些列）
    for c in CSV_OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = float("nan")

    if strict_schema:
        allowed = set(CSV_REQUIRED_COLS) | set(CSV_OPTIONAL_COLS)
        extra = [c for c in cols if c not in allowed]
        if extra:
            raise ValueError(f"CSV 出现未允许的多余列: {extra}；请删除或设 strict_schema=False")

    # NODE_ID：必须非空且唯一
    if df["NODE_ID"].isna().any():
        bad_rows = df.index[df["NODE_ID"].isna()].tolist()[:10]
        raise ValueError(f"NODE_ID 存在空值，示例行号: {bad_rows}")

    node_types = df["NODE_TYPE"].astype(str).str.strip().str.lower()
    bad_type_mask = ~node_types.isin(ALLOWED_NODE_TYPES)
    if bad_type_mask.any():
        bad = df.loc[bad_type_mask, ["NODE_ID", "NODE_TYPE"]].head(10).to_dict("records")
        raise ValueError(f"NODE_TYPE 存在非法值（只允许 {sorted(ALLOWED_NODE_TYPES)}），示例: {bad}")

    ready = pd.to_numeric(df["READY_TIME"], errors="coerce")
    due = pd.to_numeric(df["DUE_TIME"], errors="coerce")
    if ready.isna().any() or due.isna().any():
        bad_rows = df.index[ready.isna() | due.isna()].tolist()[:10]
        raise ValueError(f"READY_TIME/DUE_TIME 存在无法解析的数值，示例行号: {bad_rows}")
    bad_tw = (ready > due)
    if bad_tw.any():
        bad = df.loc[bad_tw, ["NODE_ID", "READY_TIME", "DUE_TIME"]].head(10).to_dict("records")
        raise ValueError(f"时间窗不合法（READY_TIME > DUE_TIME），示例: {bad}")

    for col in ["ORIG_X", "ORIG_Y"]:
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any():
            bad = df.index[v.isna()].tolist()[:10]
            raise ValueError(f"{col} 存在空/非法值，示例行号: {bad}")


def recompute_cost_and_nearest_base(data: Data, feasible_bases: Optional[Sequence[int]] = None) -> None:
    """重算 costMatrix，并为每个客户写回最近基站 base_id。"""
    N = len(data.nodes)
    cost = np.zeros((N, N), dtype=float)

    for i in range(N):
        xi, yi = data.nodes[i]["x"], data.nodes[i]["y"]
        for j in range(N):
            if i == j:
                cost[i, j] = 0.0
            else:
                xj, yj = data.nodes[j]["x"], data.nodes[j]["y"]
                dist = math.hypot(xi - xj, yi - yj)
                cost[i, j] = dist if dist > 1e-10 else 1e-10
    data.costMatrix = cost

    base_indices = [i for i, n in enumerate(data.nodes) if n["node_type"] == "base"]
    if data.central_idx is None:
        raise RuntimeError("data.central_idx 尚未设置，不能重算最近基站")
    if data.central_idx not in base_indices:
        base_indices.insert(0, data.central_idx)

    if feasible_bases is not None:
        feasible_set = set(int(b) for b in feasible_bases)
        base_indices = [b for b in base_indices if b in feasible_set]
        if len(base_indices) == 0:
            raise RuntimeError("feasible_bases 为空，无法为客户分配最近基站")

    for cidx in data.customer_indices:
        best_b = -1
        best_d = float("inf")
        for bidx in base_indices:
            d = float(data.costMatrix[bidx, cidx])
            if d < best_d:
                best_d = d
                best_b = bidx
        data.nodes[cidx]["base_id"] = best_b


def read_data(
    file_path: str,
    *,
    scenario: int = 1,
    perturbation_times: Optional[Sequence[float]] = None,  # 兼容旧接口，当前不在 data_io 内使用
    strict_schema: bool = True,
    drone_capacity: Optional[float] = None,
    uav_endurance: Optional[float] = None,
) -> Data:
    """
    读 CSV -> Data，并生成 costMatrix、base_id。

    重要约定：
    - 你现在的“动态位置变更”是在求解过程中改 data.nodes[i]['x','y'] 完成的；
      因此这里默认 x,y 都取 ORIG_X/ORIG_Y。
    """
    df = pd.read_csv(file_path)
    _validate_input_df(df, strict_schema=strict_schema)

    data = Data()
    data.schema_cols = CSV_REQUIRED_COLS[:]

    df["_NODE_TYPE_NORM"] = df["NODE_TYPE"].astype(str).str.strip().str.lower()

    for _, row in df.iterrows():
        node_id = int(row["NODE_ID"])
        if node_id in data.node_id_to_idx:
            raise ValueError(f"NODE_ID 重复：{node_id}（请保证唯一）")

        node_type = str(row["_NODE_TYPE_NORM"])

        node = {
            "node_id": node_id,
            "node_type": node_type,
            "base_id": -1,

            "x": float(row["ORIG_X"]),
            "y": float(row["ORIG_Y"]),

            "orig_x": float(row["ORIG_X"]),
            "orig_y": float(row["ORIG_Y"]),

            "perturbed_x": safe_float(row["PERTURBED_X"]),
            "perturbed_y": safe_float(row["PERTURBED_Y"]),

            "demand": float(row["DEMAND"]),

            "ready_time": float(row["READY_TIME"]),
            "due_time": float(row["DUE_TIME"]),
            "request_time": safe_float(row["REQUEST_TIME"]),

            "service_time": 0.0,

            "drone_capable": 1,
            "force_truck": 0,
            "base_lock": None,

            "scenario": int(scenario),
            "coord_source": "ORIG",
        }

        data.node_id_to_idx[node_id] = len(data.nodes)
        data.nodes.append(node)

    central_indices = [i for i, n in enumerate(data.nodes) if n["node_type"] == "central"]
    if len(central_indices) != 1:
        raise ValueError(f"central 节点必须且只能有 1 个；当前数量={len(central_indices)}")
    data.central_idx = central_indices[0]

    data.base_indices = [i for i, n in enumerate(data.nodes) if n["node_type"] == "base"]
    data.customer_indices = [i for i, n in enumerate(data.nodes) if n["node_type"] == "customer"]

    if data.central_idx not in data.base_indices:
        data.base_indices.insert(0, data.central_idx)

    recompute_cost_and_nearest_base(data, feasible_bases=None)

    data.UAVCustomerinDCRange = []
    if (drone_capacity is not None) and (uav_endurance is not None):
        for cidx in data.customer_indices:
            c = data.nodes[cidx]
            if c.get("demand", 0.0) <= float(drone_capacity):
                for bidx in data.base_indices:
                    dist_round = data.costMatrix[bidx, cidx] + data.costMatrix[cidx, bidx]
                    if dist_round <= float(uav_endurance):
                        data.UAVCustomerinDCRange.append(cidx)
                        break

    # 你如果不想要这个打印，直接删掉即可
    coord_stats = {}
    for node in data.nodes:
        if node["node_type"] == "customer":
            coord_stats[node.get("coord_source", "UNKNOWN")] = coord_stats.get(node.get("coord_source", "UNKNOWN"), 0) + 1
    print(f"场景{scenario}坐标使用统计: {coord_stats}")

    return data
