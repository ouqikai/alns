import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import font_manager

_CN_FONT_READY = False

def _configure_matplotlib_cn_font():
    """中文注释：设置 matplotlib 中文字体（惰性/一次性）。
    - 仅在需要绘图时调用（避免 import 阶段扫描字体导致启动变慢）。
    - 任何异常都吞掉，保证不影响求解主流程。
    """
    global _CN_FONT_READY
    if _CN_FONT_READY:
        return
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
    except Exception:
        _CN_FONT_READY = True
        return

    # 中文注释：常见中文字体候选（按优先级）
    candidates = [
        "Microsoft YaHei",      # Windows
        "SimHei",               # Windows
        "Noto Sans CJK SC",      # Linux/通用
        "PingFang SC",           # macOS
        "Arial Unicode MS",      # 旧版 macOS/Office
        "DejaVu Sans",           # matplotlib 默认（兜底）
    ]

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    # 中文注释：若找不到任何候选字体，则使用默认 DejaVu Sans（保证不报错）
    chosen = next((n for n in candidates if n in available), "DejaVu Sans")

    try:
        # matplotlib 可能在不同版本上对 font.family / font.sans-serif 的行为略有差异，这里同时设置更稳健
        mpl.rcParams["font.sans-serif"] = [chosen]
        mpl.rcParams["font.family"] = chosen
        mpl.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    except Exception:
        pass

    _CN_FONT_READY = True

def compute_global_xlim_ylim(data, reloc_radius=0.8, pad_min=5.0, step_align=10.0):
    """
    计算全局统一的 xlim/ylim（用于多个场景对比图同尺度）。
    规则：
      - 基于 data.nodes 的原始坐标
      - 额外按“最大可能扰动半径”扩一圈（不依赖具体场景是否真的扰动到边界）
      - 结果对齐到 step_align（默认 10），确保刻度整齐
    """
    import math

    xs = [float(n.get('x', 0.0)) for n in data.nodes]
    ys = [float(n.get('y', 0.0)) for n in data.nodes]

    if not xs or not ys:
        return (-10.0, 110.0), (-10.0, 110.0)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 扩边界：至少 pad_min，同时把“可能的扰动半径”算进去
    # 你的坐标系里 reloc_radius=0.8 属于同一坐标尺度，因此直接加即可
    pad = max(pad_min, float(reloc_radius) + 1.0)

    x0 = math.floor((min_x - pad) / step_align) * step_align
    x1 = math.ceil((max_x + pad) / step_align) * step_align
    y0 = math.floor((min_y - pad) / step_align) * step_align
    y1 = math.ceil((max_y + pad) / step_align) * step_align

    # 防止太窄（保持至少 50）
    if x1 - x0 < 50:
        cx = 0.5 * (x0 + x1)
        x0, x1 = cx - 25, cx + 25
    if y1 - y0 < 50:
        cy = 0.5 * (y0 + y1)
        y0, y1 = cy - 25, cy + 25

    return (float(x0), float(x1)), (float(y0), float(y1))

def _normalize_decisions_for_viz(data_cur, decisions):
    """把 decisions 规范成 visualize_truck_drone 需要的 8 元组列表。"""
    out = []
    for it in decisions:
        if isinstance(it, (list, tuple)) and len(it) == 8:
            out.append(it)
        elif isinstance(it, (list, tuple)) and len(it) == 5:
            c, dec, nx, ny, reason = it
            c = int(c)
            nid = data_cur.nodes[c]['node_id']
            ox = data_cur.nodes[c]['x']
            oy = data_cur.nodes[c]['y']
            out.append((c, nid, dec, reason, ox, oy, nx, ny))
        else:
            print("[WARN] decisions 格式异常(跳过可视化标记):", it)
    return out

def visualize_truck_drone(data, truck_route, base_to_drone_customers,
                          title="Truck + Drone Solution",
                          show_legend=True,
                          show_numbers=False,
                          xlim=None,
                          ylim=None,
                          decision_time=None,
                          truck_arrival=None,
                          drone_finish=None,
                          prefix_route=None,
                          virtual_pos=None,
                          relocation_decisions=None,
                          drone_set_before=None,
                          pad=0.0,
                          fig_size=(8, 6),
                          fig_dpi=120):
    """
    参照 GRB_dec 可视化风格（不画续航圈）：
      - 中心仓库：黄色方块
      - 基站：黄色五角星
      - 无人机客户：蓝色实心点
      - 卡车客户：蓝色空心点
      - 卡车路径：红色带箭头（单向）
      - 无人机路径：浅蓝色双向箭头
    关键修复：
      - 坐标轴/网格/刻度设置与绘图逻辑解耦：传固定 xlim/ylim 也能正常绘制
      - 不使用 tight_layout（会在不同场景下挤压 axes 导致网格视觉不一致）
      - 统一为右侧预留空间放 legend：fig.subplots_adjust(right=0.80)
      - pad：允许在固定范围基础上自动扩边界（避免每次手动改范围）
    """

    _configure_matplotlib_cn_font()

    # 中文注释：延迟导入 matplotlib（仅在绘图时加载，避免非绘图实验启动时引入 GUI/字体开销）
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

    # ---------- 1) 先得到 xlim/ylim ----------
    auto_axis = (xlim is None or ylim is None or xlim == "auto" or ylim == "auto")

    if auto_axis:
        xs = [float(n.get('x', 0.0)) for n in data.nodes]
        ys = [float(n.get('y', 0.0)) for n in data.nodes]

        # 1) 虚拟车位置（可能超出原边界）
        if virtual_pos is not None:
            try:
                xs.append(float(virtual_pos[0]))
                ys.append(float(virtual_pos[1]))
            except Exception:
                pass

        # 2) 扰动 old/new 坐标（箭头可能超出原边界）
        if relocation_decisions:
            for it in relocation_decisions:
                try:
                    if isinstance(it, (list, tuple)) and len(it) >= 8:
                        ox, oy, nx, ny = float(it[4]), float(it[5]), float(it[6]), float(it[7])
                        xs.extend([ox, nx])
                        ys.extend([oy, ny])
                except Exception:
                    continue

        min_x = min(xs) if xs else 0.0
        max_x = max(xs) if xs else 0.0
        min_y = min(ys) if ys else 0.0
        max_y = max(ys) if ys else 0.0

        span = max(max_x - min_x, max_y - min_y, 1.0)
        pad_auto = max(5.0, 0.05 * span)  # 至少留 5 个单位边距

        x0 = math.floor((min_x - pad_auto) / 10.0) * 10.0
        x1 = math.ceil((max_x + pad_auto) / 10.0) * 10.0
        y0 = math.floor((min_y - pad_auto) / 10.0) * 10.0
        y1 = math.ceil((max_y + pad_auto) / 10.0) * 10.0

        # 防止过小范围
        if x1 - x0 < 50:
            cx0 = 0.5 * (x0 + x1)
            x0, x1 = cx0 - 25, cx0 + 25
        if y1 - y0 < 50:
            cy0 = 0.5 * (y0 + y1)
            y0, y1 = cy0 - 25, cy0 + 25

        xlim = (float(x0), float(x1))
        ylim = (float(y0), float(y1))
    else:
        # 用户给定固定范围：允许统一扩边界 pad
        if pad and pad > 0:
            xlim = (float(xlim[0]) - float(pad), float(xlim[1]) + float(pad))
            ylim = (float(ylim[0]) - float(pad), float(ylim[1]) + float(pad))
        else:
            xlim = (float(xlim[0]), float(xlim[1]))
            ylim = (float(ylim[0]), float(ylim[1]))

    # ---------- 2) 无论 auto/固定，都统一设置坐标轴 + 刻度 + 网格 ----------
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

    span_xy = max(float(xlim[1] - xlim[0]), float(ylim[1] - ylim[0]))
    if span_xy <= 120:
        step = 10
    elif span_xy <= 250:
        step = 20
    elif span_xy <= 600:
        step = 50
    else:
        step = 100

    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, step))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, step))
    ax.grid(True, linestyle='--', linewidth=0.6, color='gray', alpha=0.6)

    # ---------- 3) 基本索引 ----------
    central_idx = data.central_idx
    base_indices = [i for i, n in enumerate(data.nodes) if n.get('node_type') == 'base']
    customer_indices = [i for i, n in enumerate(data.nodes) if n.get('node_type') == 'customer']

    # ---------- 4) 进度信息：已服务客户 ----------
    served_customers = set()
    if decision_time is not None:
        if truck_arrival is not None:
            for c in customer_indices:
                if truck_arrival.get(c, float('inf')) <= decision_time + 1e-9:
                    served_customers.add(c)
        if drone_finish is not None:
            for c in customer_indices:
                if drone_finish.get(c, float('inf')) <= decision_time + 1e-9:
                    served_customers.add(c)

    # ---------- 5) 前缀分割：已走过卡车边 ----------
    split_idx = None
    if decision_time is not None and prefix_route:
        last_prefix = prefix_route[-1]
        if last_prefix in truck_route:
            idxs = [k for k, v in enumerate(truck_route) if v == last_prefix]
            split_idx = idxs[-1] if idxs else None

    prev_idx = None
    next_idx = None
    if split_idx is not None:
        prev_idx = truck_route[split_idx]
        if split_idx + 1 < len(truck_route):
            next_idx = truck_route[split_idx + 1]

    # ---------- 6) 计算卡车/无人机客户集合 ----------
    drone_customers = set(c for lst in base_to_drone_customers.values() for c in lst)
    truck_customers = set(truck_route) - set(base_indices) - {central_idx}
    truck_customers -= drone_customers  # 强制互斥

    # ---------- 7) 绘制中心 & 基站 ----------
    cx, cy = data.nodes[central_idx]['x'], data.nodes[central_idx]['y']
    ax.scatter(cx, cy, marker='s', s=90, c='yellow', edgecolors='black', zorder=6)

    for b in base_indices:
        bx, by = data.nodes[b]['x'], data.nodes[b]['y']
        ax.scatter(bx, by, marker='*', s=120, c='yellow', edgecolors='black', zorder=6)

    # ---------- 8) 绘制客户点（区分已服务/未服务） ----------
    drone_unserved = [c for c in drone_customers if c in customer_indices and c not in served_customers]
    if drone_unserved:
        dx = [data.nodes[c]['x'] for c in drone_unserved]
        dy = [data.nodes[c]['y'] for c in drone_unserved]
        ax.scatter(dx, dy, c='blue', s=25, zorder=5)

    drone_served = [c for c in drone_customers if c in customer_indices and c in served_customers]
    if drone_served:
        dx = [data.nodes[c]['x'] for c in drone_served]
        dy = [data.nodes[c]['y'] for c in drone_served]
        ax.scatter(dx, dy, c='gray', s=25, alpha=0.6, zorder=5)

    truck_unserved = [c for c in truck_customers if c in customer_indices and c not in served_customers]
    if truck_unserved:
        tx = [data.nodes[c]['x'] for c in truck_unserved]
        ty = [data.nodes[c]['y'] for c in truck_unserved]
        ax.scatter(tx, ty, facecolors='none', edgecolors='blue', s=25, linewidths=1.5, zorder=5)

    truck_served = [c for c in truck_customers if c in customer_indices and c in served_customers]
    if truck_served:
        tx = [data.nodes[c]['x'] for c in truck_served]
        ty = [data.nodes[c]['y'] for c in truck_served]
        ax.scatter(tx, ty, facecolors='none', edgecolors='gray', s=25, linewidths=1.5, alpha=0.6, zorder=5)

    # ---------- 9) 位置变更对比层 ----------
    if relocation_decisions:
        def _valid(x, y):
            if x is None or y is None:
                return False
            if isinstance(x, float) and math.isnan(x):
                return False
            if isinstance(y, float) and math.isnan(y):
                return False
            return True

        drone_before = drone_set_before if drone_set_before is not None else set()

        for (cidx, nid, dec, reason, ox, oy, nx, ny) in relocation_decisions:
            if not _valid(ox, oy) or not _valid(nx, ny):
                continue

            ax.plot([ox, nx], [oy, ny], linestyle="--", color="gray", linewidth=1.5, zorder=5)

            mx, my = (ox + nx) / 2.0, (oy + ny) / 2.0
            ax.text(mx + 0.6, my + 0.6, f"{nid}", fontsize=8, color="gray", zorder=5)

            if str(dec).upper() == "ACCEPT":

                if cidx in drone_before:
                    ax.scatter([ox], [oy], s=25, c="black", edgecolors="black", linewidths=1.5, zorder=6)
                else:
                    ax.scatter([ox], [oy], s=25, facecolors="none", edgecolors="black", linewidths=1.5, zorder=6)

                is_final_truck = (cidx in truck_customers) and (cidx not in drone_customers)
                if is_final_truck:
                    ax.scatter([nx], [ny], s=25, facecolors="none", edgecolors="red", linewidths=1.5, zorder=6)
                else:
                    ax.scatter([nx], [ny], s=25, c="red", edgecolors="red", linewidths=1.0, zorder=6)
            else:
                ax.scatter([nx], [ny], s=55, marker="x", c="black", linewidths=2.2, zorder=15)

    # ---------- 10) 可选标号 ----------
    if show_numbers:
        for c in customer_indices:
            nc = data.nodes[c]
            ax.text(nc['x'] + 0.3, nc['y'] + 0.3, str(nc.get('node_id', c)), fontsize=8)

    # ---------- 11) 箭头绘制：像素级贴边 ----------
    def __marker_radius_pixels(idx):
        """
        返回指定节点 idx 在当前 axes 上的 marker 等效半径（像素）。
        与上面的 scatter s 值保持一致：
          - 中心：s=90
          - 基站：s=120
          - 其他客户：s=25
        """
        if idx == central_idx:
            s_points2 = 90
        elif idx in base_indices:
            s_points2 = 120
        else:
            s_points2 = 25

        radius_points = (np.sqrt(s_points2) / 2.0)
        dpi = fig.get_dpi()
        radius_pixels = radius_points * dpi / 110.0
        return radius_pixels

    def draw_arrow_between_indices(ax_, idx_start, idx_end,
                                  color='red', lw=1.6, style='-', double=False):
        x1, y1 = data.nodes[idx_start]['x'], data.nodes[idx_start]['y']
        x2, y2 = data.nodes[idx_end]['x'], data.nodes[idx_end]['y']

        p1_pix = ax_.transData.transform((x1, y1))
        p2_pix = ax_.transData.transform((x2, y2))

        dx_pix = p2_pix[0] - p1_pix[0]
        dy_pix = p2_pix[1] - p1_pix[1]
        dist_pix = np.hypot(dx_pix, dy_pix)
        if dist_pix < 1e-6:
            return

        r1 = __marker_radius_pixels(idx_start)
        r2 = __marker_radius_pixels(idx_end)

        ux, uy = dx_pix / dist_pix, dy_pix / dist_pix

        start_adj_pix = (p1_pix[0] + ux * r1, p1_pix[1] + uy * r1)
        end_adj_pix = (p2_pix[0] - ux * r2, p2_pix[1] - uy * r2)

        start_adj_data = ax_.transData.inverted().transform(start_adj_pix)
        end_adj_data = ax_.transData.inverted().transform(end_adj_pix)

        arr = FancyArrowPatch(start_adj_data, end_adj_data,
                              arrowstyle='-|>', color=color, linewidth=lw,
                              mutation_scale=10, linestyle=style, alpha=0.9, zorder=4)
        ax_.add_patch(arr)

        if double:
            arr2 = FancyArrowPatch(end_adj_data, start_adj_data,
                                   arrowstyle='-|>', color=color, linewidth=lw,
                                   mutation_scale=10, linestyle=style, alpha=0.9, zorder=4)
            ax_.add_patch(arr2)

    # ---------- 12) 绘制卡车路径：已走(实线) / 未走(虚线) ----------
    for i in range(len(truck_route) - 1):
        a, b = truck_route[i], truck_route[i + 1]

        # 如果存在虚拟位置，且这一条边正好是 prev->next，则先跳过，后面单独画“已走/未走”两段
        if (virtual_pos is not None and prev_idx is not None and next_idx is not None and a == prev_idx and b == next_idx):
            continue

        x1, y1 = data.nodes[a]['x'], data.nodes[a]['y']
        x2, y2 = data.nodes[b]['x'], data.nodes[b]['y']

        traveled = (split_idx is not None and i < split_idx)

        if traveled:
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=2.2, alpha=0.95, zorder=3)
            draw_arrow_between_indices(ax, a, b, color='red', lw=1.8, style='-')
        else:
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=0.8, linestyle=(0, (6, 6)), alpha=0.25, zorder=2)
            draw_arrow_between_indices(ax, a, b, color='red', lw=1.2, style='--')

    # ---------- 13) 补画：部分走过的边 + 当前卡车位置 ----------
    if virtual_pos is not None and prev_idx is not None and next_idx is not None:
        x_cur, y_cur = virtual_pos
        x_prev, y_prev = data.nodes[prev_idx]['x'], data.nodes[prev_idx]['y']
        x_next, y_next = data.nodes[next_idx]['x'], data.nodes[next_idx]['y']

        ax.plot([x_prev, x_cur], [y_prev, y_cur], color='red', linewidth=2.2, alpha=0.95, zorder=3)
        ax.plot([x_cur, x_next], [y_cur, y_next], color='red', linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.65, zorder=2)

        ax.annotate("", xy=(x_next, y_next), xytext=(x_cur, y_cur),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.0, alpha=0.45))

    # ---------- 14) 绘制无人机路径：已完成(灰色) / 未完成(浅蓝) ----------
    for b, lst in base_to_drone_customers.items():
        for c in lst:
            done = (decision_time is not None and drone_finish is not None and
                    drone_finish.get(c, float('inf')) <= decision_time + 1e-9)

            color = 'gray' if done else 'skyblue'
            alpha = 0.6 if done else 1.0
            ls = '-' if done else '--'

            bx, by = data.nodes[b]['x'], data.nodes[b]['y']
            cx2, cy2 = data.nodes[c]['x'], data.nodes[c]['y']
            ax.plot([bx, cx2], [by, cy2], color=color, linewidth=1.0, linestyle=ls, alpha=alpha, zorder=3)
            draw_arrow_between_indices(ax, b, c, color=color, lw=1.2, style='-', double=True)

    # ---------- 15) 图例 ----------
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow',
               markeredgecolor='black', markersize=9, label='中心仓库'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow',
               markeredgecolor='black', markersize=10, label='基站'),
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='white',
               markeredgecolor='blue', markersize=8, label='卡车客户'),
        Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue',
               markeredgecolor='blue', markersize=8, label='无人机客户'),
        Line2D([0], [0], color='red', lw=1.6, label='卡车路径'),
        Line2D([0], [0], color='skyblue', lw=1.2, linestyle='-', label='无人机路径'),
    ]
    if show_legend:
        ax.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 关键：不要 tight_layout（它会在不同场景下动态改变轴框尺寸，导致网格视觉不一致）
    # 统一右侧留白给 legend（所有场景一致）
    fig.subplots_adjust(right=0.80)

    # 打印图片尺寸供调试，可随时注释
    # bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # print("ax size (inch):", bbox.width, bbox.height, "figsize:", fig.get_size_inches(), "dpi:", fig.dpi)
    plt.show()

    return fig, ax
