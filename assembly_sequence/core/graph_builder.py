# graph_builder.py
# -*- coding: utf-8 -*-
"""
Brick-level weighted graph builder for assembly planning & clustering.

支持两条路径：
1) 你已经有 contacts：调用 build_graph(bricks, contacts) 直接建图；
2) 没有 contacts 但有每块砖的体素占据：先用 contacts_from_voxels 推断接触，
   再调用 build_graph_from_voxels 生成图。

节点属性（graph.nodes[n]）：
  - type, layer, pos, orient, mass, risk, grounded

边属性（graph.edges[u,v]）：
  - weight (内聚权重)
  - studs（估算 stud-tube 接触数或等价度量）
  - area（接触面积/接触格子数）
  - vertical（是否竖向支撑关系）
  - shear_cap（剪切承载近似）
  - overlap（重叠惩罚，通常为0；仅当体素推断出异常重叠时可能>0）
  - layer_gap（节点层差）

权重计算（可调超参见 build_graph 参数）：
  base = w_area*area + w_studs*studs + w_shear*shear_cap
  若 vertical: base *= (1 + w_vertical_bonus)
  base *= (1 - w_overlap_penalty*overlap_clamped)
  base *= (1 - w_risk_penalty*avg_risk)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional
import networkx as nx
import numpy as np


# ----------------------------- 数据结构 -----------------------------
@dataclass(frozen=True)
class Brick:
    id: int
    type: str                 # e.g., "1x2", "2x4"
    layer: int                # integer layer index (z-floor), 0 = ground
    pos: Tuple[int, int, int] # (x, y, z) in stud grid / voxel grid
    orient: int               # 0: along +X, 1: along +Y
    mass: float = 1.0
    risk: float = 0.0         # prior risk score in [0,1], e.g., from a risk map
    grounded: bool = False    # connected to ground

@dataclass(frozen=True)
class Contact:
    u: int
    v: int
    studs: int                # # stud-tube or equivalent measure
    area: float               # contact area proxy (# of contacting cells)
    vertical: bool            # True if u-v are in vertical support relation
    shear_cap: float = 1.0    # shear capacity proxy
    overlap: int = 0          # optional penalty if illegal overlap detected


# ----------------------------- 核心建图 -----------------------------
def build_graph(
    bricks: Iterable[Brick],
    contacts: Iterable[Contact],
    *,
    w_area: float = 1.0,
    w_studs: float = 0.6,
    w_vertical_bonus: float = 0.8,
    w_shear: float = 0.4,
    w_overlap_penalty: float = 0.5,
    w_risk_penalty: float = 0.4,
) -> nx.Graph:
    """
    用给定的 bricks 与 contacts 构建加权无向图。
    """
    G = nx.Graph()

    # --- 节点 ---
    for b in bricks:
        G.add_node(
            b.id,
            type=b.type, layer=int(b.layer), pos=tuple(b.pos), orient=int(b.orient),
            mass=float(b.mass), risk=float(b.risk), grounded=bool(b.grounded)
        )

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # --- 边 ---
    for c in contacts:
        if (c.u not in G) or (c.v not in G) or (c.u == c.v):
            continue
        r_u = float(G.nodes[c.u].get("risk", 0.0))
        r_v = float(G.nodes[c.v].get("risk", 0.0))
        avg_risk = 0.5 * (r_u + r_v)

        base = w_area * float(c.area) + w_studs * float(c.studs) + w_shear * float(c.shear_cap)
        if c.vertical:
            base *= (1.0 + w_vertical_bonus)

        weight = base
        weight *= (1.0 - w_overlap_penalty * clamp01(float(c.overlap)))
        weight *= (1.0 - w_risk_penalty * clamp01(avg_risk))
        weight = max(1e-8, float(weight))

        layer_u = int(G.nodes[c.u].get("layer", 0))
        layer_v = int(G.nodes[c.v].get("layer", 0))
        layer_gap = abs(layer_u - layer_v)

        G.add_edge(
            c.u, c.v,
            weight=weight,
            studs=int(c.studs),
            area=float(c.area),
            vertical=bool(c.vertical),
            shear_cap=float(c.shear_cap),
            overlap=int(c.overlap),
            layer_gap=layer_gap
        )
    return G


def graph_stats(G: nx.Graph) -> Dict[str, Any]:
    """Quick sanity metrics helpful in logs."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return dict(nodes=0, edges=0, avg_deg=0.0, w_sum=0.0)
    avg_deg = 2.0 * m / n
    w_sum = float(sum(d.get("weight", 1.0) for _, _, d in G.edges(data=True)))
    return dict(nodes=n, edges=m, avg_deg=avg_deg, w_sum=w_sum)


# ----------------------------- 从体素推断 Contacts（可选） -----------------------------
def contacts_from_voxels(
    brick_voxels: Dict[int, np.ndarray],
    *,
    consider_vertical: bool = True,
    consider_horizontal: bool = True,
    z_up: bool = True
) -> List[Contact]:
    """
    基于各砖体素占据推断接触（适配 6-邻接定义）。
    假设每个 np.ndarray 为同尺寸 0/1 体素网格，且不同砖不重叠（若有重叠，会记到 overlap）。

    返回 Contact 列表：
      - vertical=True：若存在 (x,y,z) 属于 u 且 (x,y,z-1) 属于 v（或反之）
      - vertical=False：若存在侧向 4-邻接（同一层 z，相邻格接触）
    studs/area 估算：
      - 竖向：按竖向接触 cell 数计入 area，并把 studs 近似为 area（如需可缩放）
      - 水平：按侧向接触 cell 数计入 area，studs 默认较小（例如 area 的 1/2）
    overlap：两个砖在同一格重叠的 cell 数（非法，作为惩罚记录）
    """
    ids = list(brick_voxels.keys())
    if not ids:
        return []

    # 预先缓存各砖的占据索引
    occ = {bid: np.asarray(brick_voxels[bid], dtype=np.uint8) for bid in ids}
    shape = next(iter(occ.values())).shape

    # 验证所有体素尺寸一致
    for a in occ.values():
        if a.shape != shape:
            raise ValueError("All brick voxel grids must share the same shape.")

    # 全局重叠检测（非法重叠越多，overlap 越大）
    stack = np.zeros(shape, dtype=np.uint16)
    for a in occ.values():
        stack += a
    # 任一位置>1 说明有重叠
    overlap_grid = np.clip(stack - 1, 0, None)

    contacts: List[Contact] = []

    # 枚举砖对（上三角）
    N = len(ids)
    for i in range(N):
        u = ids[i]
        Au = occ[u]
        for j in range(i + 1, N):
            v = ids[j]
            Av = occ[v]

            # 计算重叠
            ov = int(np.sum((Au > 0) & (Av > 0)))

            added_any = False

            # 竖向接触
            if consider_vertical:
                # u 在 v 的上方：Au(x,y,z)=1 且 Av(x,y,z-1)=1
                up_cells = 0
                if z_up:
                    Au_xyz = np.where(Au > 0)
                    for x, y, z in zip(*Au_xyz):
                        if z > 0 and Av[x, y, z - 1] > 0:
                            up_cells += 1
                else:
                    # 若坐标系相反，可相应调整
                    Au_xyz = np.where(Au > 0)
                    max_z = shape[2] - 1
                    for x, y, z in zip(*Au_xyz):
                        if z < max_z and Av[x, y, z + 1] > 0:
                            up_cells += 1

                if up_cells > 0:
                    # 近似：竖向 area = 竖向接触 cell 数；studs 直接用 area 或缩放
                    area = float(up_cells)
                    studs = int(up_cells)  # 如需可乘以比例
                    contacts.append(Contact(u=u, v=v, studs=studs, area=area, vertical=True, shear_cap=1.0, overlap=ov))
                    added_any = True

                # v 在 u 的上方（互斥时这一步通常为 0，但保留健壮性）
                down_cells = 0
                if z_up:
                    Av_xyz = np.where(Av > 0)
                    for x, y, z in zip(*Av_xyz):
                        if z > 0 and Au[x, y, z - 1] > 0:
                            down_cells += 1
                else:
                    Av_xyz = np.where(Av > 0)
                    max_z = shape[2] - 1
                    for x, y, z in zip(*Av_xyz):
                        if z < max_z and Au[x, y, z + 1] > 0:
                            down_cells += 1

                if down_cells > 0:
                    area = float(down_cells)
                    studs = int(down_cells)
                    contacts.append(Contact(u=u, v=v, studs=studs, area=area, vertical=True, shear_cap=1.0, overlap=ov))
                    added_any = True

            # 水平接触（侧向 4-邻接，同层 z）
            if consider_horizontal:
                # 统计四个方向的相邻接触
                side_cells = 0
                # 位移卷积式检查
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    shifted = np.roll(Av, shift=(dx, dy, 0), axis=(0, 1, 2))
                    # 为避免环绕贡献，边界位置置 0
                    if dx == 1:
                        shifted[0, :, :] = 0
                    elif dx == -1:
                        shifted[-1, :, :] = 0
                    if dy == 1:
                        shifted[:, 0, :] = 0
                    elif dy == -1:
                        shifted[:, -1, :] = 0
                    side_cells += int(np.sum((Au > 0) & (shifted > 0)))

                if side_cells > 0:
                    # 水平接触强度一般弱于竖向：studs 取 area 的 1/2（可调）
                    area = float(side_cells)
                    studs = int(max(1, round(side_cells * 0.5)))
                    contacts.append(Contact(u=u, v=v, studs=studs, area=area, vertical=False, shear_cap=0.8, overlap=ov))
                    added_any = True

            # 若完全没有接触但存在重叠，仍记录一条弱边以供惩罚（可选）
            if (not added_any) and ov > 0:
                contacts.append(Contact(u=u, v=v, studs=1, area=1.0, vertical=False, shear_cap=0.5, overlap=ov))

    return contacts


def build_graph_from_voxels(
    brick_meta: Dict[int, Brick],
    brick_voxels: Dict[int, np.ndarray],
    *,
    w_area: float = 1.0,
    w_studs: float = 0.6,
    w_vertical_bonus: float = 0.8,
    w_shear: float = 0.4,
    w_overlap_penalty: float = 0.5,
    w_risk_penalty: float = 0.4,
    consider_vertical: bool = True,
    consider_horizontal: bool = True
) -> nx.Graph:
    """
    一键从体素占据建图：先推断 contacts，再计算权重并返回图。
    brick_meta: {brick_id -> Brick(dataclass)}（至少需要 id/type/layer/pos/orient）
    brick_voxels: {brick_id -> 0/1 ndarray}（体素越细，接触估计越精确）
    """
    # 生成 contacts
    contacts = contacts_from_voxels(
        brick_voxels,
        consider_vertical=consider_vertical,
        consider_horizontal=consider_horizontal,
        z_up=True
    )
    bricks = [brick_meta[bid] for bid in brick_voxels.keys() if bid in brick_meta]
    return build_graph(
        bricks, contacts,
        w_area=w_area, w_studs=w_studs, w_vertical_bonus=w_vertical_bonus,
        w_shear=w_shear, w_overlap_penalty=w_overlap_penalty, w_risk_penalty=w_risk_penalty
    )


# ----------------------------- 便捷工具 -----------------------------
def voxels_from_target_and_ids(
    V_full: np.ndarray,
    brick_ids_map: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    如果你有一张与 V_full 同形状的 int32/uint16 标记图 brick_ids_map（每个体素标注其所属 brick id，
    空体素=0 或 -1），这个函数可拆分出每个 brick 的 0/1 体素占据，便于上面的 contacts_from_voxels 使用。
    返回：{brick_id -> 0/1 ndarray}
    """
    if V_full.shape != brick_ids_map.shape:
        raise ValueError("V_full and brick_ids_map must share the same shape.")
    ids = np.unique(brick_ids_map)
    ids = [int(i) for i in ids if int(i) > 0]
    out: Dict[int, np.ndarray] = {}
    for bid in ids:
        mask = (brick_ids_map == bid).astype(np.uint8)
        # 仅保留目标占据内的体素
        out[bid] = (mask & (V_full > 0)).astype(np.uint8)
    return out