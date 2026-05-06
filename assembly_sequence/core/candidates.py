# candidates.py
# -*- coding: utf-8 -*-
"""
Candidate generator with physics-aware prefilters and cluster-interface support.

功能要点
- 砖库与尺寸：与用户提供一致（1x1, 1x2, 1x4, 1x6, 1x8, 2x2, 2x4, 2x6）
- 模板缓存：按 (brick_type, orient) 生成占据模板，复用加速
- 表面 Anchor 采样：仅在“有下方支撑/地面”的位置采样
- 物理预筛：
    * 边界检查（不出格）
    * 与 V_current 不碰撞（几何级）
    * 新增体素需落在 V_target 内（任务级）
    * 支撑面积比例 >= 阈值（support_ratio_th）
- 聚类接口候选：
    * 基于 G(partition) 的聚类质心，沿两团连线附近 & “need” 区域采样 anchor
- 可视化辅助：把若干候选渲染到体素叠加图里（debug 用）

注意
- orient: 0 沿 +X，1 沿 +Y（传统 LEGO 平放两种朝向）。如需 4 向旋转，可在 ORIENTS 拓展到 {0,1,2,3} 并在模板计算里加旋转。
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Any, Optional
from functools import lru_cache
from collections import defaultdict

import numpy as np

from .interface_mask import Action

# ---------------------------------------------------------------------
# 砖库（与用户提供一致）
# ---------------------------------------------------------------------
BRICK_LIBRARY = {
    "1x1": "3005.dat",
    "1x2": "3004.dat",
    "1x4": "3010.dat",
    "1x6": "3009.dat",
    "1x8": "3008.dat",
    "2x2": "3003.dat",
    "2x4": "3001.dat",
    "2x6": "2456.dat",
}

# (width_in_studs_along_Y, length_in_studs_along_X, height_layers)
# 这里采用约定：orient=0 时占据 (lenX × widY × 1)；orient=1 时交换 XY。
BRICK_DIMS = {
    "1x1": (1, 1, 1),
    "1x2": (1, 2, 1),
    "1x4": (1, 4, 1),
    "1x6": (1, 6, 1),
    "1x8": (1, 8, 1),
    "2x2": (2, 2, 1),
    "2x4": (2, 4, 1),
    "2x6": (2, 6, 1),
}

ORIENTS = (0, 1)  # 0: +X, 1: +Y

# ---------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------
def _in_bounds(x: int, y: int, z: int, shape: Tuple[int, int, int]) -> bool:
    nx, ny, nz = shape
    return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)

def _clip_add(V: np.ndarray, add: np.ndarray) -> np.ndarray:
    return np.clip(V + add, 0, 1).astype(V.dtype)

def _support_ratio(V_current: np.ndarray, add: np.ndarray) -> float:
    """
    计算新增体素底层面的支撑比例：
    ratio = 支撑体素数 / 新增体素总数
    支撑体素定义：新增体素的 (x,y,z) 若 z>0 且 (x,y,z-1) 在 V_current 中为 1；
                 或 z==0（地面）默认视为支撑。
    """
    xs, ys, zs = np.where(add > 0)
    if xs.size == 0:
        return 0.0
    sup = 0
    for i in range(xs.size):
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        if z == 0:
            sup += 1
        else:
            if V_current[x, y, z - 1] == 1:
                sup += 1
    return float(sup) / float(xs.size)

def _touches_existing(V_current: np.ndarray, add: np.ndarray) -> bool:
    """
    判断新增体素是否与现有结构有至少一个 6-邻接（侧/上/下直接相邻）。
    避免“完全漂浮只靠上空接触”的候选（更强的几何预筛）。
    """
    xs, ys, zs = np.where(add > 0)
    if xs.size == 0:
        return False
    nx, ny, nz = V_current.shape
    for i in range(xs.size):
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            xx, yy, zz = x+dx, y+dy, z+dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                if V_current[xx, yy, zz] == 1:
                    return True
    return False

# ---------------------------------------------------------------------
# 模板缓存
# ---------------------------------------------------------------------
@lru_cache(maxsize=64)
def _brick_template(brick_type: str, orient: int) -> np.ndarray:
    """
    生成砖在局部坐标系 (0..size_x-1, 0..size_y-1, z=0) 的占据模板。
    """
    if brick_type not in BRICK_DIMS:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    wid_y, len_x, h = BRICK_DIMS[brick_type]
    if orient == 0:  # 沿 +X 放置
        size_x, size_y = len_x, wid_y
    else:            # 沿 +Y 放置
        size_x, size_y = wid_y, len_x
    tmpl = np.zeros((size_x, size_y, h), dtype=np.uint8)
    tmpl[:, :, 0] = 1
    return tmpl

def _place_template(anchor: Tuple[int, int, int], tmpl: np.ndarray, grid_shape: Tuple[int,int,int]) -> Optional[np.ndarray]:
    """
    将模板体素放置到全局体素网格，返回同形状的 0/1 体素。如果越界，返回 None。
    """
    x0, y0, z0 = anchor
    size_x, size_y, size_z = tmpl.shape
    nx, ny, nz = grid_shape
    x1, y1, z1 = x0 + size_x, y0 + size_y, z0 + size_z
    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > nx or y1 > ny or z1 > nz:
        return None
    V = np.zeros(grid_shape, dtype=np.uint8)
    V[x0:x1, y0:y1, z0:z1] = tmpl
    return V

# ---------------------------------------------------------------------
# Anchor 采样（表面/need/接口）
# ---------------------------------------------------------------------
def find_surface_anchors(
    V_current: np.ndarray,
    *,
    stride_xy: int = 1,
    z_from: int = 0,
    z_to: Optional[int] = None
) -> List[Tuple[int,int,int]]:
    """
    在当前表面采样 anchor:
      条件：z>0 且 V_cur[x,y,z]==0 且 V_cur[x,y,z-1]==1
           或 z==0 且 V_cur[x,y,0]==0（地面）
    """
    anchors: List[Tuple[int,int,int]] = []
    nx, ny, nz = V_current.shape
    zz_to = nz if z_to is None else min(nz, z_to)
    for z in range(max(0, z_from), zz_to):
        for x in range(0, nx, max(1, stride_xy)):
            for y in range(0, ny, max(1, stride_xy)):
                if V_current[x, y, z] != 0:
                    continue
                if z == 0 or V_current[x, y, z-1] == 1:
                    anchors.append((x, y, z))
    return anchors

def find_need_anchors(
    V_current: np.ndarray,
    V_target: np.ndarray,
    *,
    stride_xy: int = 1
) -> List[Tuple[int,int,int]]:
    """
    在 need 区域 (V_target - V_current)>0 采样 anchor（不考虑支撑）
    """
    need = (V_target - V_current) > 0
    xs, ys, zs = np.where(need)
    pts = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    if stride_xy > 1:
        pts = pts[::stride_xy]
    return pts

def _cluster_centroid_xy_from_graph(G, part: Dict[int,int], cluster_id: int) -> Optional[Tuple[float, float, float]]:
    """
    由图 G 的节点属性 pos=(x,y,z)、layer 估计某团的 (cx,cy,avg_layer)。
    若缺属性则返回 None。
    """
    xs, ys, ls = [], [], []
    for n, data in G.nodes(data=True):
        if part.get(n, -1) != cluster_id:
            continue
        pos = data.get("pos", None)
        layer = data.get("layer", None)
        if pos is None or layer is None:
            continue
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        ls.append(float(layer))
    if not xs:
        return None
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    avg_l = float(np.mean(ls)) if ls else 0.0
    return (cx, cy, avg_l)

def find_interface_anchors(
    G, part: Dict[int,int],
    u_cluster: int, v_cluster: int,
    V_current: np.ndarray, V_target: np.ndarray,
    *,
    band_halfwidth: float = 4.0,
    stride_xy: int = 1
) -> List[Tuple[int,int,int]]:
    """
    在两个子块的“连线带状区域”内、且属于 need 区域的体素位置采样 anchor。
    这是几何近似：我们仅用聚类质心（由 brick graph 节点 pos/layer 估计）来定位接口区域。
    """
    cu = _cluster_centroid_xy_from_graph(G, part, u_cluster)
    cv = _cluster_centroid_xy_from_graph(G, part, v_cluster)
    if cu is None or cv is None:
        # 回退：直接用 need anchors
        return find_need_anchors(V_current, V_target, stride_xy=stride_xy)

    (x1, y1, _), (x2, y2, _) = cu, cv
    need = (V_target - V_current) > 0
    xs, ys, zs = np.where(need)
    anchors: List[Tuple[int,int,int]] = []
    # 线段 (x1,y1)-(x2,y2) 到点 (x,y) 的距离是否在带宽内
    vx, vy = (x2 - x1), (y2 - y1)
    vv = vx*vx + vy*vy + 1e-9
    for (x, y, z) in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        # 投影点
        t = ((x - x1)*vx + (y - y1)*vy) / vv
        t = max(0.0, min(1.0, t))
        px, py = (x1 + t*vx), (y1 + t*vy)
        dist = ((x - px)**2 + (y - py)**2)**0.5
        if dist <= band_halfwidth:
            anchors.append((x, y, z))
    if stride_xy > 1:
        anchors = anchors[::stride_xy]
    return anchors

# ---------------------------------------------------------------------
# 生成候选（内层/接口）
# ---------------------------------------------------------------------
def _enumerate_actions_at_anchor(
    anchor: Tuple[int,int,int],
    brick_types: Iterable[str],
    allow_orients: Iterable[int],
    grid_shape: Tuple[int,int,int]
) -> List[Action]:
    acts: List[Action] = []
    for bt in brick_types:
        if bt not in BRICK_DIMS:
            continue
        for o in allow_orients:
            tmpl = _brick_template(bt, o)
            placed = _place_template(anchor, tmpl, grid_shape)
            if placed is None:
                continue
            acts.append(Action(bt, anchor, o, placed))
    return acts

def _prefilter_action(
    a: Action,
    V_current: np.ndarray,
    V_target: np.ndarray,
    *,
    require_inside_target: bool = True,
    require_no_collision: bool = True,
    require_support_ratio: float = 0.25,
    require_touch_existing: bool = True,
) -> bool:
    """
    物理/几何预筛（快速）：
      - 新增体素是否全部在目标内
      - 与当前结构不重叠
      - 底部支撑比例阈值
      - 至少与现有结构 6-邻接一次（或落地）
    """
    add = a.vox
    if require_inside_target and not np.all(add <= V_target):
        return False
    if require_no_collision and not np.all((add + V_current) <= 1):
        return False
    if require_support_ratio is not None and require_support_ratio > 0:
        if _support_ratio(V_current, add) < float(require_support_ratio):
            return False
    if require_touch_existing and not _touches_existing(V_current, add):
        # 如果直接放在地面（z==0），也允许
        if not np.any(np.where(add > 0)[2] == 0):
            return False
    return True

def propose_candidates_in_cluster(
    cluster_nodes: List[int],   # 当前子块节点（可不使用，保留接口）
    G,
    V_current: np.ndarray,
    V_target: np.ndarray,
    inventory: Dict[str, int],
    *,
    anchor_mode: str = "surface",   # "surface" | "need"
    stride_xy: int = 1,
    allow_orients: Iterable[int] = ORIENTS,
    brick_types: Iterable[str] = tuple(BRICK_LIBRARY.keys()),
    max_candidates: int = 512,
    require_support_ratio: float = 0.25,
    require_touch_existing: bool = True,
) -> List[Action]:
    """
    子块内候选生成：
      - 默认在“表面 anchor”采样（有下方支撑）
      - 对每个 anchor × (brick, orient) 组合生成占据
      - 进行物理预筛
      - 受 inventory 控制
    """
    if anchor_mode == "surface":
        anchors = find_surface_anchors(V_current, stride_xy=stride_xy)
    else:
        anchors = find_need_anchors(V_current, V_target, stride_xy=stride_xy)

    cand: List[Action] = []
    for anchor in anchors:
        acts = _enumerate_actions_at_anchor(anchor, brick_types, allow_orients, V_current.shape)
        for a in acts:
            if inventory.get(a.brick_type, 0) <= 0:
                continue
            if not _prefilter_action(
                a, V_current, V_target,
                require_inside_target=True,
                require_no_collision=True,
                require_support_ratio=require_support_ratio,
                require_touch_existing=require_touch_existing,
            ):
                continue
            cand.append(a)
            if len(cand) >= max_candidates:
                return cand
    return cand

def propose_candidates_for_interface(
    cluster_id: int,
    neighbor_id: int,
    G,
    part: Dict[int,int],
    V_current: np.ndarray,
    V_target: np.ndarray,
    inventory: Dict[str, int],
    *,
    band_halfwidth: float = 4.0,
    stride_xy: int = 1,
    allow_orients: Iterable[int] = ORIENTS,
    brick_types: Iterable[str] = tuple(BRICK_LIBRARY.keys()),
    max_candidates: int = 256,
    require_support_ratio: float = 0.20,  # 接口处可稍微放宽
    require_touch_existing: bool = True,
) -> List[Action]:
    """
    子块对接候选：
      - 在两个子块质心连线的带状区域内，且属于 need 区域的位置采样 anchor
      - 生成 (brick × orient) 候选
      - 物理预筛（可稍放宽支撑占比，以便桥接）
    """
    anchors = find_interface_anchors(
        G, part, cluster_id, neighbor_id, V_current, V_target,
        band_halfwidth=band_halfwidth, stride_xy=stride_xy
    )
    cand: List[Action] = []
    for anchor in anchors:
        acts = _enumerate_actions_at_anchor(anchor, brick_types, allow_orients, V_current.shape)
        for a in acts:
            if inventory.get(a.brick_type, 0) <= 0:
                continue
            if not _prefilter_action(
                a, V_current, V_target,
                require_inside_target=True,
                require_no_collision=True,
                require_support_ratio=require_support_ratio,
                require_touch_existing=require_touch_existing,
            ):
                continue
            cand.append(a)
            if len(cand) >= max_candidates:
                return cand
    return cand

# ---------------------------------------------------------------------
# 可视化辅助（调试用）：把候选渲染到一个叠加体素图里
# ---------------------------------------------------------------------
def visualize_candidates(
    V_current: np.ndarray,
    candidates: List[Action],
    max_show: int = 32
) -> np.ndarray:
    """
    返回一个体素叠加图（uint8）：
      0 = 空；1 = 当前结构；2..255 = 叠加的候选（最多 max_show 个）
    注意：仅用于快速预览/调试。
    """
    out = V_current.astype(np.uint8).copy()
    k = 0
    for a in candidates[:max_show]:
        k = min(250, k + 1)
        add = a.vox
        xs, ys, zs = np.where(add > 0)
        for i in range(xs.size):
            x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
            if _in_bounds(x, y, z, out.shape) and out[x, y, z] == 0:
                out[x, y, z] = 2 + (k % 254)
    return out