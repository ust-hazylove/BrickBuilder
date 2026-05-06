# ldr_stability.py
# -*- coding: utf-8 -*-
"""
Brick-level stability solver for LDraw (Y-up), no voxelization.
- Parse .ldr directly, snap X/Z on 0.5-stud grid, continuous Y (supports brick/plate/tile).
- Per-(x,z) half-grid vertical "interval pairing" (with ground) + same-layer horizontals.
- Physics:
    * Vertical compression N>=0; tangential shear Sx,Sz with |Sx|+|Sz| <= μN + C0 (brick-brick / ground).
    * Horizontal edges Fx,Fz (optional caps).
    * Per-brick L1 residuals on ΣFx, ΣFy-W, ΣFz, τx, τz.
- Ground options:
    * --ground-rigid : ground is rigid (no N cap, no Coulomb limit at ground).
- Y-axis options:
    * --flip-y-physics : mirror Y in PHYSICS (default ON). Geometry is flipped before building contacts.
- Viz:
    * --viz-mode brick|grid, --viz-azim/--viz-elev,
      --viz-swap-xz/--viz-mirror-x/--viz-mirror-z/--viz-mirror-y (purely visual).
"""

from __future__ import annotations
import argparse, os, sys, glob, math, csv
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from gurobipy import Model, GRB, quicksum

# ---- matplotlib (headless) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Units & constants ----------------
LDU_PER_STUD   = 20
LDU_PER_BRICK  = 24
LDU_PER_PLATE  = 8
HALFGRID       = 2           # 0.5-stud grid
G              = 9.81        # m/s^2

# ---------------- Size & mass ----------------
# (rows, cols) = (along Z studs, along X studs)
LDPART_TO_SIZE: Dict[str, Tuple[int, int]] = {
    # Bricks
    "3005.dat": (1, 1), "3004.dat": (1, 2), "3010.dat": (1, 4),
    "3009.dat": (1, 6), "3008.dat": (1, 8), "3003.dat": (2, 2),
    "3001.dat": (2, 4), "2456.dat": (2, 6),
    # Plates 1xN
    "3024.dat": (1,1), "3023.dat": (1,2), "3623.dat": (1,3),
    "3710.dat": (1,4), "3666.dat": (1,6), "3460.dat": (1,8),
    # Plates 2xN
    "3022.dat": (2,2), "3021.dat": (2,3), "3020.dat": (2,4),
    "3795.dat": (2,6), "3034.dat": (2,8),
    # Tiles (use plate footprint)
    "3070b.dat": (1,1), "3069b.dat": (1,2), "3068b.dat": (2,2),
}
FOOTPRINT_MASS = {
    (1, 1): 0.00043, (1, 2): 0.00081, (1, 4): 0.00157,
    (1, 6): 0.00228, (1, 8): 0.00303, (2, 2): 0.00115,
    (2, 4): 0.00216, (2, 6): 0.00323,
}
PART_HEIGHT_BRICK = {
    # bricks
    "3005.dat": 1.0, "3004.dat": 1.0, "3010.dat": 1.0, "3009.dat": 1.0,
    "3008.dat": 1.0, "3003.dat": 1.0, "3001.dat": 1.0, "2456.dat": 1.0,
    # plates / tiles
    "3024.dat": 1.0/3, "3023.dat": 1.0/3, "3623.dat": 1.0/3,
    "3710.dat": 1.0/3, "3666.dat": 1.0/3, "3460.dat": 1.0/3,
    "3022.dat": 1.0/3, "3021.dat": 1.0/3, "3020.dat": 1.0/3,
    "3795.dat": 1.0/3, "3034.dat": 1.0/3,
    "3070b.dat": 1.0/3, "3069b.dat": 1.0/3, "3068b.dat": 1.0/3,
}
FALLBACK_SIZE = (1,1)
FALLBACK_MASS = 0.001

# ---------------- Data ----------------
@dataclass
class Brick:
    idx: int
    part: str
    rows: int; cols: int
    mass: float
    x0_h: int; y0_b: float; z0_h: int
    h_b: float

@dataclass
class Interval:
    yb: float; yt: float; bid: int

# ---------------- Helpers ----------------
def nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rr = U @ Vt
    if np.linalg.det(Rr) < 0:
        U[:, -1] *= -1
        Rr = U @ Vt
    return Rr

def classify_upright_y(R: np.ndarray, y_relax_rad: float, accept_inverted=True):
    Rr = nearest_rotation_matrix(R)
    ycol = Rr[:,1]
    cos_plus = float(np.clip(ycol @ np.array([0,1,0], float), -1, 1))
    ang_plus = math.acos(cos_plus)
    Rz = Rr
    if ang_plus > y_relax_rad:
        cos_minus = float(np.clip(ycol @ np.array([0,-1,0], float), -1, 1))
        ang_minus = math.acos(cos_minus)
        if accept_inverted and ang_minus <= y_relax_rad:
            Ry_pi = np.array([[-1,0,0],[0,1,0],[0,0,-1]], float)
            Rz = Rr @ Ry_pi
        else:
            return (False, None)
    a,c = float(Rz[0,0]), float(Rz[0,2])
    theta = math.atan2(c, a)
    k = int(round(theta / (math.pi/2)))
    snapped_deg = (k % 4) * 90
    if abs(theta - k*(math.pi/2)) > y_relax_rad:
        return (False, None)
    return (True, snapped_deg)

def snap_to_halfstud(v_stud: float) -> int:
    return int(round(v_stud * HALFGRID))

# ---------------- Parse LDR ----------------
def parse_ldr_to_bricks(path: str, y_relax_deg: float, force_upright: bool,
                        accept_inverted: bool, keep_unknown=True, verbose=True) -> List[Brick]:
    y_relax_rad = math.radians(y_relax_deg)
    bricks: List[Brick] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f,1):
            line=line.strip()
            if not line or line.startswith("0"): continue
            toks=line.split()
            if toks[0]!="1" or len(toks)<15: continue
            x, y, z = map(float, toks[2:5])
            a,b,c,d,e,ff,g,h,i = map(float, toks[5:14])
            part = toks[14].lower()

            if part in LDPART_TO_SIZE:
                rows, cols = LDPART_TO_SIZE[part]
                height_b = PART_HEIGHT_BRICK.get(part, 1.0)
                mass = FOOTPRINT_MASS.get((rows, cols), FALLBACK_MASS)
            else:
                if not keep_unknown:
                    if verbose: print(f"[WARN] skip unknown part {part} @ L{ln}")
                    continue
                rows, cols = FALLBACK_SIZE
                height_b = PART_HEIGHT_BRICK.get(part, 1.0) if part in PART_HEIGHT_BRICK else 1.0
                mass = FALLBACK_MASS
                if verbose: print(f"[WARN] unknown part {part} -> fallback {rows}x{cols}, mass={mass} kg")

            R = np.array([[a,b,c],[d,e,ff],[g,h,i]], float)
            ok, ydeg = classify_upright_y(R, y_relax_rad, accept_inverted)
            if not ok:
                if force_upright:
                    ydeg = 0
                    if verbose: print(f"[WARN] force upright {part} at L{ln}")
                else:
                    if verbose: print(f"[WARN] non-upright skipped {part} at L{ln}")
                    continue
            if ydeg in (90,270):
                rows, cols = cols, rows

            cx_h = snap_to_halfstud(x / LDU_PER_STUD)
            cz_h = snap_to_halfstud(z / LDU_PER_STUD)
            x0_h = cx_h - (cols*HALFGRID // 2)
            z0_h = cz_h - (rows*HALFGRID // 2)
            y0_b = (y / LDU_PER_BRICK)

            bricks.append(Brick(
                idx=len(bricks), part=part, rows=rows, cols=cols, mass=mass,
                x0_h=x0_h, y0_b=y0_b, z0_h=z0_h, h_b=height_b
            ))
    if not bricks:
        raise RuntimeError(f"{path}: no supported parts parsed.")
    return bricks

# ---------------- Y flip in PHYSICS ----------------
def apply_y_flip_in_physics(bricks: List[Brick], enable: bool=True):
    """Mirror Y before building contacts. Keep gravity sign/constraints unchanged."""
    if not enable or not bricks:
        return
    # mirror around the global top surface
    y_top_all = max(b.y0_b + b.h_b for b in bricks)
    for b in bricks:
        b.y0_b = y_top_all - (b.y0_b + b.h_b)  # mirror
    # shift so minimum becomes 0
    y_min = min(b.y0_b for b in bricks)
    for b in bricks:
        b.y0_b -= y_min

# ---------------- World building ----------------
def resolve_overlaps(intervals: List[Interval]) -> List[Interval]:
    intervals.sort(key=lambda t: (t.yb, t.yt))
    out: List[Interval] = []
    for seg in intervals:
        if not out: out.append(seg); continue
        if seg.yb < out[-1].yt - 1e-6:
            if (seg.yt - seg.yb) < (out[-1].yt - out[-1].yb):
                out[-1] = seg
        else:
            out.append(seg)
    return out

def build_world_grid(bricks: List[Brick], y_tol: float = 1.0/6.0):
    occ_half: Dict[Tuple[int,int], List[Interval]] = {}
    for b in bricks:
        for xx_h in range(b.x0_h, b.x0_h + b.cols*HALFGRID):
            for zz_h in range(b.z0_h, b.z0_h + b.rows*HALFGRID):
                key = (xx_h, zz_h)
                occ_half.setdefault(key, []).append(Interval(b.y0_b, b.y0_b + b.h_b, b.idx))

    vert_pairs: List[Tuple[Tuple[int,int,float], Optional[Tuple[int,int,float]], bool]] = []
    for key, ivs in list(occ_half.items()):
        ivs = resolve_overlaps(ivs); ivs.sort(key=lambda t: t.yb); occ_half[key] = ivs
        if not ivs: continue
        # ground
        vert_pairs.append(((key[0], key[1], ivs[0].yb), None, True))
        # brick-brick
        for i in range(len(ivs)-1):
            a, b = ivs[i], ivs[i+1]
            if abs(a.yt - b.yb) <= y_tol:
                vert_pairs.append(((key[0], key[1], b.yb), (key[0], key[1], a.yb), False))

    # Horizontal neighbors (coarse layer index)
    horiz_slices: Dict[Tuple[int,int,int], int] = {}
    for (x_h, z_h), ivs in occ_half.items():
        for it in ivs:
            y_layer = int(round((it.yb + it.yt) * 0.5))
            horiz_slices[(x_h, z_h, y_layer)] = it.bid
    return occ_half, horiz_slices, vert_pairs

def find_bid_at(occ_half: Dict[Tuple[int,int], List[Interval]], x_h: int, z_h: int, y_b: float) -> int:
    ivs = occ_half.get((x_h, z_h))
    if not ivs: raise KeyError(f"No intervals at ({x_h},{z_h})")
    for it in ivs:
        if (it.yb - 1e-6) <= y_b <= (it.yt + 1e-6) or abs(it.yb - y_b) <= 1e-6:
            return it.bid
    return min(ivs, key=lambda t: abs(t.yb - y_b)).bid

# ---------------- Solver ----------------
def build_and_solve(
    bricks: List[Brick],
    occ_half, horiz_slices, vert_pairs,
    cap_per_stud: float, shear_cap_per_edge: float,
    mu_vert: float, c0_vert: float, mu_ground: float, c0_ground: float,
    alpha_reg: float, beta_reg: float,
    ground_rigid: bool = False,
    verbose=True,
    # === 新增的三个改进的控制参数（有默认值，可不传） ===
    edge_shear_frac: float = 0.20,   # 边界可承受剪力比例（相对 shear_cap_per_edge）
    Mx_cap: float = 0.60,            # 每砖绕X的跨越弯矩容量上限
    Mz_cap: float = 0.60,            # 每砖绕Z的跨越弯矩容量上限
    align_radius_stud: int = 1       # 竖向点位“就近对齐”的搜索半径（单位：stud）
):
    m = Model("ldr_stability_b_model+edge+M+align")
    m.Params.OutputFlag = 1 if verbose else 0
    try: m.Params.NonConvex = 2
    except Exception: pass

    V = list(range(len(vert_pairs)))
    B_idx = list(range(len(bricks)))

    # ---- 砖心（stud 坐标）----
    b_center = {}
    for b in bricks:
        cx_stud = (b.x0_h + b.cols * HALFGRID / 2.0) / HALFGRID
        cz_stud = (b.z0_h + b.rows * HALFGRID / 2.0) / HALFGRID
        cy_brick = b.y0_b + b.h_b / 2.0
        b_center[b.idx] = (cx_stud, cz_stud, cy_brick)

    # ---- 横向边（相邻才建边）----
    H_list: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = []
    Fx = []; Fz = []
    for (x_h,z_h,y_l), _ in horiz_slices.items():
        nb = (x_h+1, z_h, y_l)
        if nb in horiz_slices:
            H_list.append(((x_h,z_h,y_l), nb))
            Fx.append(m.addVar(lb=-GRB.INFINITY)); Fz.append(m.addVar(lb=-GRB.INFINITY))
        nb = (x_h, z_h+1, y_l)
        if nb in horiz_slices:
            H_list.append(((x_h,z_h,y_l), nb))
            Fx.append(m.addVar(lb=-GRB.INFINITY)); Fz.append(m.addVar(lb=-GRB.INFINITY))

    # 原有的剪切上限（用于相邻边）
    if shear_cap_per_edge and shear_cap_per_edge > 0:
        for j in range(len(H_list)):
            m.addConstr(Fx[j] <=  shear_cap_per_edge); m.addConstr(Fx[j] >= -shear_cap_per_edge)
            m.addConstr(Fz[j] <=  shear_cap_per_edge); m.addConstr(Fz[j] >= -shear_cap_per_edge)

    # === 改进 1：边界给一点剪力 ===
    T_edge_small = (edge_shear_frac * shear_cap_per_edge) if (shear_cap_per_edge and edge_shear_frac>0) else 0.0
    # 对每个 cell，若右/前方向没有邻居，则给“边界剪力”变量（仅施加到本砖）
    EdgeFx = {}; EdgeFz = {}
    if T_edge_small > 0:
        for (x_h,z_h,y_l), bid in horiz_slices.items():
            # 右侧邻居缺失 → 允许少量 Fx
            if (x_h+1, z_h, y_l) not in horiz_slices:
                vx = m.addVar(lb=-T_edge_small, ub=+T_edge_small, name=f"edge_Fx_x+_xh{x_h}_zh{z_h}_yl{y_l}")
                EdgeFx[(x_h,z_h,y_l,"x+")] = vx
            if (x_h-1, z_h, y_l) not in horiz_slices:
                vx = m.addVar(lb=-T_edge_small, ub=+T_edge_small, name=f"edge_Fx_x-_xh{x_h}_zh{z_h}_yl{y_l}")
                EdgeFx[(x_h,z_h,y_l,"x-")] = vx
            # 前后邻居缺失 → 允许少量 Fz
            if (x_h, z_h+1, y_l) not in horiz_slices:
                vz = m.addVar(lb=-T_edge_small, ub=+T_edge_small, name=f"edge_Fz_z+_xh{x_h}_zh{z_h}_yl{y_l}")
                EdgeFz[(x_h,z_h,y_l,"z+")] = vz
            if (x_h, z_h-1, y_l) not in horiz_slices:
                vz = m.addVar(lb=-T_edge_small, ub=+T_edge_small, name=f"edge_Fz_z-_xh{x_h}_zh{z_h}_yl{y_l}")
                EdgeFz[(x_h,z_h,y_l,"z-")] = vz

    # ---- 顶/底水平挤压 + 上下点接触 ----
    force_dict: Dict[Tuple[int,int,int], Dict[str, any]] = {}
    def ensure(key, name, lb=0.0):
        if key is None:
            raise ValueError(f"ensure() got None key for var '{name}'")
        d = force_dict.setdefault(key, {})
        if name not in d:
            d[name] = m.addVar(lb=lb, name=f"{name}_xh{key[0]}_zh{key[1]}_yl{key[2]}")
        return d[name]

    is_four_pt = {b.idx: (min(b.cols, b.rows) < 2) for b in bricks}

    # === 改进 3：就近对齐：若底部 cell 无支撑，搜索最近支撑 cell（≤ align_radius_stud studs）===
    def has_support_at(xh:int, zh:int, yb:float) -> bool:
        ivs = occ_half.get((xh,zh)); 
        if not ivs: return False
        for it in ivs:
            if (it.yb - 1e-6) <= yb <= (it.yt + 1e-6) or abs(it.yb - yb) <= 1e-6:
                return True
        return False
    from collections import deque

    def occupied_at_layer(yb):
        """把同层的占用单元做成一个 set，便于 O(1) 查询。"""
        return {(xh, zh) for (xh, zh, yl) in horiz_slices.keys() if yl == yb}
    

    def path_exists_same_layer(start_cell, goal_cell, max_offset_half):
        """
        同层 BFS：只能沿“有砖的单元”走（不穿越空气）。
        start_cell/goal_cell: (xh, zh, yb)
        max_offset_half: 搜索边界（半格为单位），与就近半径一致
        """
        x0, z0, yb = start_cell
        xg, zg, _  = goal_cell
        occ = occupied_at_layer(yb)
        if (xg, zg) not in occ:
            return False  # 目标必须是有砖

        # 起点如果是空气，允许从其 4 邻的第一圈有砖单元起步
        starts = []
        if (x0, z0) in occ:
            starts = [(x0, z0)]
        else:
            for dx, dz in ((1,0),(-1,0),(0,1),(0,-1)):
                p = (x0+dx, z0+dz)
                if p in occ:
                    starts.append(p)
        if not starts:
            return False

        # 有界 BFS（限制盒子，避免跑太远）
        xmin, xmax = x0 - max_offset_half, x0 + max_offset_half
        zmin, zmax = z0 - max_offset_half, z0 + max_offset_half

        Q = deque(starts)
        seen = set(starts)
        while Q:
            x, z = Q.popleft()
            if (x, z) == (xg, zg):
                return True
            for dx, dz in ((1,0),(-1,0),(0,1),(0,-1)):  # 4 邻接
                xn, zn = x+dx, z+dz
                if not (xmin <= xn <= xmax and zmin <= zn <= zmax):
                    continue
                if (xn, zn) not in occ:
                    continue
                if (xn, zn) in seen:
                    continue
                seen.add((xn, zn))
                Q.append((xn, zn))
        return False

    def nearest_supported_cell(bottom_cell_raw):
        """
        结构化“桥接”条件（方案 C）：
        只有当上层下投点到底层候选支撑之间，存在同层“砖-砖连通链”（不穿越空气），
        才允许就近对齐；否则返回原位（视为悬臂，不对齐）。
        """
        xh, zh, yb = bottom_cell_raw

        # 正下方有支撑 → 直接用原位
        if has_support_at(xh, zh, yb):
            return bottom_cell_raw

        # 就近搜索半径（stud → halfgrid）
        max_offset_half = align_radius_stud * HALFGRID
        if max_offset_half <= 0:
            return bottom_cell_raw  # 不允许对齐

        # 收集半径内所有“有支撑”的候选
        candidates = []
        for dx in range(-max_offset_half, max_offset_half+1):
            for dz in range(-max_offset_half, max_offset_half+1):
                if dx == 0 and dz == 0:
                    continue
                xn, zn = xh + dx, zh + dz
                if has_support_at(xn, zn, yb):
                    candidates.append((xn, zn, yb, dx*dx + dz*dz))

        # 只保留与下投点存在“同层连通链”的候选，并取最近的一个
        for xn, zn, yb1, d2 in sorted(candidates, key=lambda t: t[3]):
            if path_exists_same_layer((xh, zh, yb), (xn, zn, yb1), max_offset_half):
                return (xn, zn, yb1)

        # 没有桥接路径 → 禁止就近对齐（悬臂）
        return bottom_cell_raw


    # 顶/底配对
    def link_top_bottom(up_cell, dn_cell):
        txp = ensure(up_cell, "top_x_pos"); txn = ensure(up_cell, "top_x_neg")
        tzp = ensure(up_cell, "top_z_pos"); tzn = ensure(up_cell, "top_z_neg")
        bxp = ensure(dn_cell, "bottom_x_pos"); bxn = ensure(dn_cell, "bottom_x_neg")
        bzp = ensure(dn_cell, "bottom_z_pos"); bzn = ensure(dn_cell, "bottom_z_neg")
        m.addConstr(txp == bxn); m.addConstr(txn == bxp)
        m.addConstr(tzp == bzn); m.addConstr(tzn == bzp)

    # 竖向对 + 地面
    FN = []
    for i,(up, dn, is_ground) in enumerate(vert_pairs):
        up_bid = find_bid_at(occ_half, up[0], up[1], up[2])
        npt_up = 4 if is_four_pt.get(up_bid, True) else 3

        if is_ground:
            # ← 修复：贴地单元就是 up
            bottom_cell = up
            # 底部水平挤压（与地面）
            ensure(bottom_cell, "bottom_x_pos"); ensure(bottom_cell, "bottom_x_neg")
            ensure(bottom_cell, "bottom_z_pos"); ensure(bottom_cell, "bottom_z_neg")
            # 点接触：由这块砖 footprint 决定 3/4 点
            npt = 4 if is_four_pt.get(up_bid, True) else 3
            f_down = [m.addVar(lb=0.0, name=f"f_down_i{i}_k{k}") for k in range(npt)]
            n_up   = [m.addVar(lb=0.0, name=f"n_up_i{i}_k{k}")   for k in range(npt)]
            for k in range(npt):
                m.addQConstr(f_down[k] * n_up[k] == 0.0)
            FN.append(("ground", bottom_cell, f_down, n_up))
            continue

        # 砖-砖
        top_cell = up
        # === 改进 3：竖向点位对齐到底层最近可支撑的 cell ===
        bottom_cell_raw = dn
        bottom_cell = nearest_supported_cell(bottom_cell_raw)

        # 顶/底水平挤压配对保持原几何（仍然用原来的 bottom_cell_raw）
        ensure(top_cell, "top_x_pos"); ensure(top_cell, "top_x_neg")
        ensure(top_cell, "top_z_pos"); ensure(top_cell, "top_z_neg")
        ensure(bottom_cell_raw, "bottom_x_pos"); ensure(bottom_cell_raw, "bottom_x_neg")
        ensure(bottom_cell_raw, "bottom_z_pos"); ensure(bottom_cell_raw, "bottom_z_neg")
        link_top_bottom(top_cell, bottom_cell_raw)

        # 上簇 f_up/n_down
        f_up   = [m.addVar(lb=0.0, name=f"f_up_i{i}_k{k}")   for k in range(npt_up)]
        n_down = [m.addVar(lb=0.0, name=f"n_down_i{i}_k{k}") for k in range(npt_up)]
        for k in range(npt_up):
            m.addQConstr(f_up[k] * n_down[k] == 0.0)

        # 下簇 f_down/n_up（由底砖 footprint 决定 3/4 点；施加在 bottom_cell（已就近对齐）上）
        dn_bid = find_bid_at(occ_half, bottom_cell[0], bottom_cell[1], bottom_cell[2])
        npt_dn = 4 if is_four_pt.get(dn_bid, True) else 3
        f_down = [m.addVar(lb=0.0, name=f"f_down_i{i}_k{k}") for k in range(npt_dn)]
        n_up   = [m.addVar(lb=0.0, name=f"n_up_i{i}_k{k}")   for k in range(npt_dn)]
        for k in range(npt_dn):
            m.addQConstr(f_down[k] * n_up[k] == 0.0)

        # 同序对齐（若点数不同，取 min）
        for k in range(min(npt_up, npt_dn)):
            m.addConstr(f_up[k]   == f_down[k])
            m.addConstr(n_down[k] == n_up[k])

        FN.append((top_cell, bottom_cell, f_up, n_down, f_down, n_up))

    # ---- 累加器 ----
    Fx_sum={b:[] for b in B_idx}; Fy_sum={b:[] for b in B_idx}; Fz_sum={b:[] for b in B_idx}
    Tx_sum={b:[] for b in B_idx}; Tz_sum={b:[] for b in B_idx}

    # 1) 同层邻接边
    for j,(a,bk) in enumerate(H_list):
        (x1_h,z1_h,y1),(x2_h,z2_h,y2) = a,bk
        bid1 = horiz_slices[(x1_h,z1_h,y1)]
        bid2 = horiz_slices[(x2_h,z2_h,y2)]
        if x2_h == x1_h + 1 and z1_h == z2_h:
            Fx_sum[bid1].append((+1.0, Fx[j])); Fx_sum[bid2].append((-1.0, Fx[j]))
            z_cell1=(z1_h+0.5)/HALFGRID; z_cell2=(z2_h+0.5)/HALFGRID
            Tz_sum[bid1].append(((z_cell1-b_center[bid1][1]), Fx[j]))
            Tz_sum[bid2].append((-(z_cell2-b_center[bid2][1]), Fx[j]))
        elif z2_h == z1_h + 1 and x1_h == x2_h:
            Fz_sum[bid1].append((+1.0, Fz[j])); Fz_sum[bid2].append((-1.0, Fz[j]))
            x_cell1=(x1_h+0.5)/HALFGRID; x_cell2=(x2_h+0.5)/HALFGRID
            Tx_sum[bid1].append((-(x_cell1-b_center[bid1][0]), Fz[j]))
            Tx_sum[bid2].append(( +(x_cell2-b_center[bid2][0]), Fz[j]))

    # 1b) === 改进 1：边界剪力施加到本砖 ===
    if T_edge_small > 0:
        for (x_h,z_h,y_l), bid in horiz_slices.items():
            # Fx 边界
            for tag in ("x+","x-"):
                key=(x_h,z_h,y_l,tag)
                if key in EdgeFx:
                    Fx_sum[bid].append((+1.0, EdgeFx[key]))
                    # 以砖高/2 作为臂长映射到绕 Z 的力矩（与上面邻接边一致）
                    Tz_sum[bid].append((+0.5, EdgeFx[key]))
            # Fz 边界
            for tag in ("z+","z-"):
                key=(x_h,z_h,y_l,tag)
                if key in EdgeFz:
                    Fz_sum[bid].append((+1.0, EdgeFz[key]))
                    Tx_sum[bid].append((+0.5, EdgeFz[key]))

    # 2) 顶/底 挤压 + 点接触
    OFF4  = [+0.25, 0.0, -0.25, 0.0]
    OFF4X = [0.0, -0.25, 0.0, +0.25]
    OFF3  = [+0.125, 0.0, -0.125]
    OFF3X = [+0.125, -0.25, +0.125]
    unit_h = 1.0; unit_l = 1.0

    for item in FN:
        if item[0] == "ground":
            bottom_cell = item[1]; f_down, n_up = item[2], item[3]
            bd = find_bid_at(occ_half, bottom_cell[0], bottom_cell[1], bottom_cell[2])

            bx_pos = ensure(bottom_cell,"bottom_x_pos"); bx_neg = ensure(bottom_cell,"bottom_x_neg")
            bz_pos = ensure(bottom_cell,"bottom_z_pos"); bz_neg = ensure(bottom_cell,"bottom_z_neg")
            Fx_sum[bd].append((+1.0, bx_pos)); Fx_sum[bd].append((-1.0, bx_neg))
            Fz_sum[bd].append((+1.0, bz_pos)); Fz_sum[bd].append((-1.0, bz_neg))
            Tx_sum[bd].append(( +unit_h/2.0, bz_pos)); Tx_sum[bd].append(( -unit_h/2.0, bz_neg))
            Tz_sum[bd].append(( +unit_h/2.0, bx_neg)); Tz_sum[bd].append(( -unit_h/2.0, bx_pos))

            npt=len(f_down); offY = OFF4 if npt==4 else OFF3; offX = OFF4X if npt==4 else OFF3X
            for k in range(npt):
                Fz_sum[bd].append((+1.0, n_up[k])); Fz_sum[bd].append((-1.0, f_down[k]))
                Tz_sum[bd].append(( +(offX[k])*unit_l, n_up[k])); Tz_sum[bd].append(( -(offX[k])*unit_l, f_down[k]))
                Tx_sum[bd].append(( +(offY[k])*unit_l, n_up[k])); Tx_sum[bd].append(( -(offY[k])*unit_l, f_down[k]))
            continue

        top_cell, bottom_cell, f_up, n_down, f_down, n_up = item
        bu = find_bid_at(occ_half, top_cell[0], top_cell[1], top_cell[2])
        bd = find_bid_at(occ_half, bottom_cell[0], bottom_cell[1], bottom_cell[2])

        # 顶/底水平挤压（用原几何 bottom_cell_raw 已在上面配对，这里直接拿变量）
        txp = ensure(top_cell,"top_x_pos"); txn = ensure(top_cell,"top_x_neg")
        tzp = ensure(top_cell,"top_z_pos"); tzn = ensure(top_cell,"top_z_neg")
        bxp = ensure((bottom_cell[0], bottom_cell[1], bottom_cell[2]),"bottom_x_pos")
        bxn = ensure((bottom_cell[0], bottom_cell[1], bottom_cell[2]),"bottom_x_neg")
        bzp = ensure((bottom_cell[0], bottom_cell[1], bottom_cell[2]),"bottom_z_pos")
        bzn = ensure((bottom_cell[0], bottom_cell[1], bottom_cell[2]),"bottom_z_neg")

        Fx_sum[bu].append((+1.0, txp)); Fx_sum[bu].append((-1.0, txn))
        Fz_sum[bu].append((+1.0, tzp)); Fz_sum[bu].append((-1.0, tzn))
        Tx_sum[bu].append(( +unit_h/2.0, tzp)); Tx_sum[bu].append(( -unit_h/2.0, tzn))
        Tz_sum[bu].append(( +unit_h/2.0, txn)); Tz_sum[bu].append(( -unit_h/2.0, txp))

        Fx_sum[bd].append((+1.0, bxp)); Fx_sum[bd].append((-1.0, bxn))
        Fz_sum[bd].append((+1.0, bzp)); Fz_sum[bd].append((-1.0, bzn))
        Tx_sum[bd].append(( +unit_h/2.0, bzp)); Tx_sum[bd].append(( -unit_h/2.0, bzn))
        Tz_sum[bd].append(( +unit_h/2.0, bxn)); Tz_sum[bd].append(( -unit_h/2.0, bxp))

        npt_u=len(f_up); offYu = OFF4 if npt_u==4 else OFF3; offXu = OFF4X if npt_u==4 else OFF3X
        for k in range(npt_u):
            Fz_sum[bu].append((+1.0, f_up[k])); Fz_sum[bu].append((-1.0, n_down[k]))
            Tx_sum[bu].append(( +(offYu[k])*unit_l, f_up[k])); Tx_sum[bu].append(( -(offYu[k])*unit_l, n_down[k]))
            Tz_sum[bu].append(( -(offXu[k])*unit_l, f_up[k])); Tz_sum[bu].append(( +(offXu[k])*unit_l, n_down[k]))

        npt_d=len(f_down); offYd = OFF4 if npt_d==4 else OFF3; offXd = OFF4X if npt_d==4 else OFF3X
        for k in range(npt_d):
            Fz_sum[bd].append((+1.0, n_up[k])); Fz_sum[bd].append((-1.0, f_down[k]))
            Tx_sum[bd].append(( +(offYd[k])*unit_l, n_up[k])); Tx_sum[bd].append(( -(offYd[k])*unit_l, f_down[k]))
            Tz_sum[bd].append(( +(offXd[k])*unit_l, n_up[k])); Tz_sum[bd].append(( -(offXd[k])*unit_l, f_down[k]))

    # ---- 残差变量 + 弯矩（改进 2）----
    rpx=m.addVars(B_idx,lb=0); rnx=m.addVars(B_idx,lb=0)
    rpy=m.addVars(B_idx,lb=0); rny=m.addVars(B_idx,lb=0)
    rpz=m.addVars(B_idx,lb=0); rnz=m.addVars(B_idx,lb=0)
    tpx=m.addVars(B_idx,lb=0); tnx=m.addVars(B_idx,lb=0)
    tpz=m.addVars(B_idx,lb=0); tnz=m.addVars(B_idx,lb=0)

    # 每砖弯矩自由度（跨越能力）
    Mx = {b: m.addVar(lb=-Mx_cap, ub=+Mx_cap, name=f"Mx_b{b}") for b in B_idx}
    Mz = {b: m.addVar(lb=-Mz_cap, ub=+Mz_cap, name=f"Mz_b{b}") for b in B_idx}

    obj_terms=[]
    for b in B_idx:
        W = bricks[b].mass * G
        sumFx = quicksum(c*v for c,v in Fx_sum[b]) if Fx_sum[b] else 0.0
        sumFy = quicksum(c*v for c,v in Fy_sum[b]) if Fy_sum[b] else 0.0
        sumFz = quicksum(c*v for c,v in Fz_sum[b]) if Fz_sum[b] else 0.0
        sumTx = quicksum(c*v for c,v in Tx_sum[b]) if Tx_sum[b] else 0.0
        sumTz = quicksum(c*v for c,v in Tz_sum[b]) if Tz_sum[b] else 0.0

        m.addConstr(sumFx      == rpx[b]-rnx[b])
        m.addConstr(sumFy      == rpy[b]-rny[b])
        m.addConstr(sumFz - W  == rpz[b]-rnz[b])
        # === 改进 2：弯矩参与平衡 ===
        m.addConstr(sumTx + Mx[b] == tpx[b]-tnx[b])
        m.addConstr(sumTz + Mz[b] == tpz[b]-tnz[b])

        obj_terms += [rpx[b],rnx[b],rpy[b],rny[b],rpz[b],rnz[b],tpx[b],tnx[b],tpz[b],tnz[b]]

    # α*Σ max(f_down) + β*Σ f_up（保留）
    # 同时记录 f_up 以便 β 正则
    sum_f_up_terms=[]
    for item in FN:
        if item[0] == "ground":  # ground 只有 f_down/n_up
            continue
        _,_, f_up, _, _, _ = item
        sum_f_up_terms += list(f_up)

    if alpha_reg>0:
        # 每砖下压的最大值
        brick_f_down_buckets = {b:[] for b in B_idx}
        for item in FN:
            if item[0] == "ground":
                bottom_cell, f_down = item[1], item[2]
                bd = find_bid_at(occ_half, bottom_cell[0], bottom_cell[1], bottom_cell[2])
                brick_f_down_buckets[bd].extend(f_down)
            else:
                _, bottom_cell, _, _, f_down, _ = item
                bd = find_bid_at(occ_half, bottom_cell[0], bottom_cell[1], bottom_cell[2])
                brick_f_down_buckets[bd].extend(f_down)
        for b in B_idx:
            if brick_f_down_buckets[b]:
                mdown = m.addVar(lb=0.0, name=f"brick{b}_max_fdown")
                for fd in brick_f_down_buckets[b]:
                    m.addConstr(mdown >= fd)
                obj_terms.append(alpha_reg * mdown)

    if beta_reg>0 and sum_f_up_terms:
        obj_terms.append(beta_reg * quicksum(sum_f_up_terms))

    m.setObjective(quicksum(obj_terms), GRB.MINIMIZE)
    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi status={m.Status}")

    # ---- 风险计算（不变）----
    stud_len = 1.0
    risk = np.zeros(len(bricks))
    for b in B_idx:
        f_abs=(rpx[b].X+rnx[b].X)+(rpy[b].X+rny[b].X)+(rpz[b].X+rnz[b].X)
        t_abs=(tpx[b].X+tnx[b].X)+(tpz[b].X+tnz[b].X)
        risk[b]=f_abs+t_abs/stud_len

    # 导出（与原版一致）
    vert_stats=[]; horiz_stats=[]; brick_stats=[]
    for i,(up, dn, is_ground) in enumerate(vert_pairs):
        bu = find_bid_at(occ_half, up[0], up[1], up[2])
        bd = -1 if is_ground else find_bid_at(occ_half, dn[0], dn[1], dn[2])
        vert_stats.append({"k":i,"upper_brick":bu,"lower_brick":(bd if not is_ground else -1),"is_ground":int(is_ground)})
    for j,(a,bk) in enumerate(H_list):
        ba=horiz_slices[a]; bb=horiz_slices[bk]
        horiz_stats.append({"k":j,"brick_a":ba,"brick_b":bb,"Fx":Fx[j].X,"Fz":Fz[j].X})
    for b in B_idx:
        bb=bricks[b]
        brick_stats.append({"brick_id":b,"part":bb.part,"x0_half":bb.x0_h,"y0_brick":round(bb.y0_b,4),
                            "z0_half":bb.z0_h,"rows":bb.rows,"cols":bb.cols,"height_b":bb.h_b,
                            "mass_kg":bb.mass,"risk":float(risk[b])})
    return risk, brick_stats, vert_stats, horiz_stats


# ---------------- Visualization ----------------
def _apply_axes_options(ax, bricks, swap_xz: bool, mirror_x: bool, mirror_z: bool,
                        elev: float, azim: float, mirror_y: bool=False):
    xs=[b.x0_h/HALFGRID for b in bricks]; zs=[b.z0_h/HALFGRID for b in bricks]; ys=[b.y0_b for b in bricks]
    x_max=max(x + b.cols for x,b in zip(xs,bricks)); z_max=max(z + b.rows for z,b in zip(zs,bricks)); y_max=max(y + b.h_b for y,b in zip(ys,bricks))
    x_min=min(xs); z_min=min(zs); y_min=max(0, min(ys))
    ax.set_box_aspect(((z_max-z_min) if swap_xz else (x_max-x_min),
                       (x_max-x_min) if swap_xz else (z_max-z_min),
                       (y_max-y_min)))
    if swap_xz:
        ax.set_xlabel('Z (stud)'); ax.set_ylabel('X (stud)')
    else:
        ax.set_xlabel('X (stud)'); ax.set_ylabel('Z (stud)')
    ax.set_zlabel('Y (brick)')
    ax.view_init(elev=elev, azim=azim)
    if mirror_x:
        if swap_xz: ax.invert_yaxis()
        else:       ax.invert_xaxis()
    if mirror_z:
        if swap_xz: ax.invert_xaxis()
        else:       ax.invert_yaxis()
    if mirror_y:
        ax.invert_zaxis()   # purely visual

def save_brick_visual(bricks, risk, thresh, path_png,
                      swap_xz=False, mirror_x=False, mirror_z=False, mirror_y=False,
                      elev=25, azim=-60):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    for b in bricks:
        rgba=(1,0,0,0.6) if (risk[b.idx] > thresh) else (1,1,1,1.0)
        x0=b.x0_h/HALFGRID; z0=b.z0_h/HALFGRID; y0=b.y0_b
        dx=b.cols; dz=b.rows; dy=b.h_b
        if swap_xz:
            x0, z0 = z0, x0
            dx, dz = dz, dx
        ax.bar3d(x0, z0, y0, dx, dz, dy, shade=False,
                 color=rgba, edgecolor=(0,0,0,0.25), linewidth=0.3)
    _apply_axes_options(ax, bricks, swap_xz, mirror_x, mirror_z, elev, azim, mirror_y)
    ax.set_title('LDR Stability (white=stable, red=high-risk)')
    plt.tight_layout(); plt.savefig(path_png, dpi=300); plt.close(fig)

def save_voxel_visual(bricks, risk, thresh, path_png,
                      swap_xz=False, mirror_x=False, mirror_z=False, mirror_y=False,
                      elev=25, azim=-60):
    max_xh=max(b.x0_h + b.cols*HALFGRID for b in bricks)
    max_zh=max(b.z0_h + b.rows*HALFGRID for b in bricks)
    max_y=int(max(b.y0_b + b.h_b for b in bricks) + 1.0)
    occ=np.zeros((max_xh, max_zh, max_y), dtype=np.uint8)
    for b in bricks:
        color=2 if risk[b.idx]>thresh else 1; y_layer=int(round(b.y0_b))
        for xx_h in range(b.x0_h, b.x0_h + b.cols*HALFGRID):
            for zz_h in range(b.z0_h, b.z0_h + b.rows*HALFGRID):
                occ[xx_h, zz_h, y_layer]=color
    vox=np.transpose(occ>0, (2,1,0)); base=np.transpose(occ,(2,1,0))
    if swap_xz:
        vox=np.transpose(vox, (0,2,1)); base=np.transpose(base,(0,2,1))
    rgba=np.zeros(vox.shape+(4,), float)
    rgba[vox & (base==1)] = (1,1,1,1.0)
    rgba[vox & (base==2)] = (1,0,0,0.6)
    fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111, projection='3d')
    ax.voxels(vox, facecolors=rgba, edgecolor='k', linewidth=0.1)
    _apply_axes_options(ax, bricks, swap_xz, mirror_x, mirror_z, elev, azim, mirror_y)
    ax.set_title('LDR Stability (white=stable, red=high-risk)')
    plt.tight_layout(); plt.savefig(path_png, dpi=300); plt.close(fig)

# ---------------- IO ----------------
def write_csv(path: str, rows: List[dict], header: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader()
        for r in rows: w.writerow(r)

# ---------------- Pipeline ----------------
def process_one(path_ldr: str, out_dir: str,
                y_relax_deg: float, accept_inverted: bool, force_upright: bool,
                cap_per_stud: float, shear_cap: float,
                mu_vert: float, c0_vert: float, mu_ground: float, c0_ground: float,
                alpha_reg: float, beta_reg: float,
                risk_thresh: float, keep_unknown: bool,
                viz_mode: str, viz_swap_xz: bool, viz_mirror_x: bool, viz_mirror_z: bool, viz_mirror_y: bool,
                viz_elev: float, viz_azim: float,
                ground_rigid: bool, flip_y_physics: bool,
                save_vis: bool, save_csv: bool, verbose=True):
    name=os.path.splitext(os.path.basename(path_ldr))[0]
    bricks=parse_ldr_to_bricks(path_ldr, y_relax_deg, force_upright, accept_inverted, keep_unknown, verbose)
    # <<<<<< flip Y in PHYSICS >>>>>>
    apply_y_flip_in_physics(bricks, enable=flip_y_physics)

    occ_half, horiz_slices, vert_pairs=build_world_grid(bricks)
    if verbose:
        print(f"[INFO] {name}: bricks={len(bricks)} horiz_edges≈{len(horiz_slices)} vert_contacts={len(vert_pairs)}")

    risk, brick_stats, vert_stats, horiz_stats=build_and_solve(
        bricks, occ_half, horiz_slices, vert_pairs,
        cap_per_stud, shear_cap, mu_vert, c0_vert, mu_ground, c0_ground,
        alpha_reg, beta_reg, ground_rigid=ground_rigid, verbose=verbose)

    if risk_thresh is None or risk_thresh < 0:
        med=float(np.median(risk)); mad=float(np.median(np.abs(risk-med))+1e-9)
        risk_thresh=med+3.0*1.4826*mad

    os.makedirs(out_dir, exist_ok=True)
    if save_vis:
        png=os.path.join(out_dir, f"{name}_stability.png")
        if viz_mode=="grid":
            save_voxel_visual(bricks, risk, risk_thresh, png,
                              viz_swap_xz, viz_mirror_x, viz_mirror_z, viz_mirror_y,
                              viz_elev, viz_azim)
        else:
            save_brick_visual(bricks, risk, risk_thresh, png,
                              viz_swap_xz, viz_mirror_x, viz_mirror_z, viz_mirror_y,
                              viz_elev, viz_azim)
        print(f"[OK] vis -> {png}")

    if save_csv:
        bcsv=os.path.join(out_dir, f"{name}_brick_stats.csv")
        vcsv=os.path.join(out_dir, f"{name}_vertical_contacts.csv")
        hcsv=os.path.join(out_dir, f"{name}_horizontal_edges.csv")
        write_csv(bcsv, brick_stats, ["brick_id","part","x0_half","y0_brick","z0_half","rows","cols","height_b","mass_kg","risk"])
        write_csv(vcsv, vert_stats,  ["k","upper_brick","lower_brick","is_ground","N","Sx_abs","Sz_abs"])
        write_csv(hcsv, horiz_stats, ["k","brick_a","brick_b","Fx","Fz"])
        print(f"[OK] csv -> {bcsv}\n[OK] csv -> {vcsv}\n[OK] csv -> {hcsv}")

def main():
    ap=argparse.ArgumentParser(description="Brick-level stability solver for LDraw (Y-up), half-stud & plates supported")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out_stability")
    # Pose
    ap.add_argument("--y-relax-deg", type=float, default=6.0)
    ap.add_argument("--accept-inverted", action="store_true", default=True)
    ap.add_argument("--force-upright", action="store_true")
    # Physics
    ap.add_argument("--cap-per-stud", type=float, default=12.0)
    ap.add_argument("--shear-cap", type=float, default=4.0)
    ap.add_argument("--mu-vert", type=float, default=0.35)
    ap.add_argument("--c0-vert", type=float, default=0.25)
    ap.add_argument("--mu-ground", type=float, default=0.45)
    ap.add_argument("--c0-ground", type=float, default=0.50)
    ap.add_argument("--alpha-reg", type=float, default=0.0)
    ap.add_argument("--beta-reg", type=float, default=0.0)
    ap.add_argument("--ground-rigid", action="store_true",
                    help="ground rigid: no N cap & no Coulomb limit at ground")
    ap.add_argument("--flip-y-physics", action="store_true", default=True,
                    help="mirror Y in PHYSICS before contact building (default ON)")
    # Viz
    ap.add_argument("--viz-mode", choices=["brick","grid"], default="brick")
    ap.add_argument("--viz-azim", type=float, default=-60.0)
    ap.add_argument("--viz-elev", type=float, default=25.0)
    ap.add_argument("--viz-swap-xz", action="store_true")
    ap.add_argument("--viz-mirror-x", action="store_true")
    ap.add_argument("--viz-mirror-z", action="store_true")
    ap.add_argument("--viz-mirror-y", action="store_true")
    # Risk & IO
    ap.add_argument("--risk-thresh", type=float, default=0.05)
    ap.add_argument("--keep-unknown", action="store_true", default=True)
    ap.add_argument("--save-vis", action="store_true")
    ap.add_argument("--save-csv", action="store_true")

    args=ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.input, "*.ldr"))) if os.path.isdir(args.input) else [args.input]
    if not paths: print("[ERR] no .ldr found"); sys.exit(1)

    for p in paths:
        process_one(p, args.out,
            args.y_relax_deg, args.accept_inverted, args.force_upright,
            args.cap_per_stud, args.shear_cap, args.mu_vert, args.c0_vert, args.mu_ground, args.c0_ground,
            args.alpha_reg, args.beta_reg,
            args.risk_thresh, args.keep_unknown,
            args.viz_mode, args.viz_swap_xz, args.viz_mirror_x, args.viz_mirror_z, args.viz_mirror_y,
            args.viz_elev, args.viz_azim,
            args.ground_rigid, args.flip_y_physics,
            args.save_vis, args.save_csv, verbose=True)

if __name__ == "__main__":
    main()