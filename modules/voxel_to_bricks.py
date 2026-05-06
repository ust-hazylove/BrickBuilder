# -*- coding: utf-8 -*-
"""
voxel_to_bricks.py

用法：
python batch_voxel_to_bricks_v3.py --input-dir <dir> --output-dir <dir> \
  --input-format npy --make-plots --export-1x1-ldr [--invert-z]
"""

import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# ========== 基础结构 ==========
Brick = namedtuple("Brick", ["type","x","y","z","dx","dy","rot","color"])

# LDraw 基本砖（Brick，不是 Plate）
LDRAW_PARTS = {
     
}

# 合并策略：先两行矩形（稳且省砖），再单行长条，最后 1x1
PAIR_LIBRARY = [(2,6),(2,4),(2,2)]    # dy=2, dx=6/4/2
ROW_LIBRARY  = [(1,8),(1,6),(1,4),(1,2)]
FALLBACK     = [(1,1)]

# ========== 工具 ==========
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_voxels(path, input_format):
    arr = np.load(path)
    if input_format == "npy":
        if arr.ndim != 3:
            raise ValueError(f"{path}: expected 3D array for 'npy'")
        V = arr.astype(bool)
        xs, ys, zs = np.nonzero(V)
        coords = np.stack([xs, ys, zs], 1)
    elif input_format == "coords":
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{path}: expected Nx3 integer coords")
        coords = arr.astype(int)
    else:
        raise ValueError("input-format must be 'npy' or 'coords'")
    if coords.size == 0:
        raise ValueError(f"{path}: empty voxels")
    return coords

def transform_coords(coords):
    # (x_new, y_new, z_new) = (x_old, z_old, y_old)
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    new = np.stack([x, z, y], 1)
    # 平移到起点为 0
    new = new - new.min(0, keepdims=True)
    return new.astype(int)

def coords_to_grid(coords):
    maxs = coords.max(0) + 1
    V = np.zeros((maxs[0], maxs[1], maxs[2]), dtype=bool)  # V[x,y,z]
    V[coords[:,0], coords[:,1], coords[:,2]] = True
    return V

def visualize_coords(coords, save_path=None, title="", show=False):
    xs,ys,zs = coords[:,0], coords[:,1], coords[:,2]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title); ax.set_box_aspect([1,1,1])
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

# ========== 先 1×1 后合并 ==========
def one_by_one_bricks(V):
    """每个占据体素变成 1x1 砖（完整覆盖）"""
    bricks=[]
    X,Y,Z = V.shape
    for z in range(Z):
        layer = V[:,:,z]
        xs, ys = np.nonzero(layer.T)  # .T: layer.T[y,x] = V[x,y,z]
        for x,y in zip(xs,ys):
            bricks.append(Brick("1x1", int(x), int(y), int(z), 1,1,0,16))
    return bricks

def try_rect(layer_used, layer_occ, x0,y0, dx,dy):
    H,W = layer_occ.shape
    if x0+dx>W or y0+dy>H: return False
    if not layer_occ[y0:y0+dy, x0:x0+dx].all(): return False
    if layer_used[y0:y0+dy, x0:x0+dx].any(): return False
    return True

def mark_used(layer_used, x0,y0, dx,dy):
    layer_used[y0:y0+dy, x0:x0+dx] = True

def merge_layer(layer_occ):
    """
    输入：layer_occ[y,x] 布尔图（该层的占据格）
    输出：该层砖列表（完全覆盖 layer_occ=True）
    """
    H,W = layer_occ.shape
    used = np.zeros_like(layer_occ, bool)
    bricks=[]

    # 1) 两行矩形（2×6/2×4/2×2）
    for y in range(0, H-1):
        x=0
        while x < W:
            placed=False
            for (dy,dx) in PAIR_LIBRARY:  # dy=2
                if dy!=2: continue
                if try_rect(used, layer_occ, x,y, dx,dy):
                    mark_used(used, x,y, dx,dy)
                    bricks.append(Brick(f"{min(2,dy)}x{max(2,dx)}", x,y,0, dx,dy, 0,16))
                    x += dx
                    placed=True
                    break
            if not placed: x += 1

    # 2) 单行长条（1×8/1×6/1×4/1×2）
    for y in range(H):
        x=0
        while x < W:
            if not layer_occ[y,x] or used[y,x]:
                x += 1; continue
            placed=False
            for (dy,dx) in ROW_LIBRARY:  # dy=1
                if dy!=1: continue
                if try_rect(used, layer_occ, x,y, dx,dy):
                    mark_used(used, x,y, dx,dy)
                    bricks.append(Brick(f"{min(1,dy)}x{max(1,dx)}", x,y,0, dx,dy, 0,16))
                    x += dx
                    placed=True
                    break
            if not placed: x += 1

    # 3) 兜底 1×1
    yy,xx = np.where(layer_occ & (~used))
    for x,y in zip(xx,yy):
        mark_used(used, x,y, 1,1)
        bricks.append(Brick("1x1", x,y,0, 1,1,0,16))

    # 覆盖校验
    if used.sum() != layer_occ.sum():
        # 极端情况下补齐
        yy,xx = np.where(layer_occ & (~used))
        for x,y in zip(xx,yy):
            mark_used(used, x,y, 1,1)
            bricks.append(Brick("1x1", x,y,0, 1,1,0,16))
    return bricks

def merge_all_layers_from_grid(V):
    """V[x,y,z] → 合并后的 Brick 列表（不跨层）"""
    X,Y,Z = V.shape
    merged=[]
    for z in range(Z):
        layer = V[:,:,z].T               # layer[y,x]
        if not layer.any(): continue
        bs = merge_layer(layer)
        for b in bs:
            merged.append(Brick(b.type, b.x, b.y, z, b.dx, b.dy, b.rot, b.color))
    return merged

# ========== 导出 ==========
def write_json(bricks, out_json):
    ensure_dir(os.path.dirname(out_json))
    payload=[]
    for b in bricks:
        payload.append({
            "type": b.type,
            "pos":  [int(b.x), int(b.y), int(b.z)],   # 左下角(格) + 层
            "size": [int(b.dx), int(b.dy)],          # XY 覆盖
            "rot":  int(b.rot),                      # 0/90
            "color":int(b.color),
        })
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)

def rotz(deg):
    d=deg%360
    if d==0:   return (1,0,0, 0,1,0, 0,0,1)
    if d==90:  return (0,-1,0, 1,0,0, 0,0,1)
    if d==180: return (-1,0,0, 0,-1,0, 0,0,1)
    if d==270: return (0,1,0, -1,0,0, 0,0,1)
    raise ValueError("rotation must be multiple of 90°")

def flip_upside():  # 绕 X 轴 180°：把砖从“倒扣”翻为“stud 朝 +Y”
    return (1,0,0, 0,-1,0, 0,0,-1)

def matmul3(m1, m2):
    a1,b1,c1, d1,e1,f1, g1,h1,i1 = m1
    a2,b2,c2, d2,e2,f2, g2,h2,i2 = m2
    return (
        a1*a2+b1*d2+c1*g2, a1*b2+b1*e2+c1*h2, a1*c2+b1*f2+c1*i2,
        d1*a2+e1*d2+f1*g2, d1*b2+e1*e2+f1*h2, d1*c2+e1*f2+f1*i2,
        g1*a2+h1*d2+i1*g2, g1*b2+h1*e2+i1*h2, g1*c2+h1*f2+i1*i2
    )

def write_ldr(bricks, out_ldr, invert_z=False, stud_spacing=20, brick_height=24):
    """
    LDraw 轴：X 左右，Y 向上(=stud方向)，Z 前后
    - 坐标映射：
        X = (x + dx/2) * stud_spacing
        Z = (y + dy/2) * stud_spacing   # 默认正方向；如需反向传入 invert_z=True
        Y = (z + 0.5) * brick_height
    - 朝向矩阵：Rz(rot) @ Rx(180°)  —— 保证 stud 朝 +Y
    """
    ensure_dir(os.path.dirname(out_ldr))
    with open(out_ldr,"w",encoding="utf-8") as f:
        f.write("0 Generated by batch_voxel_to_bricks_v3.py\n")
        for b in bricks:
            part = LDRAW_PARTS.get(b.type,"3005.dat")
            X = (b.x + b.dx/2.0) * stud_spacing
            Z = (b.y + b.dy/2.0) * stud_spacing
            if invert_z: Z = -Z
            Y = (b.z + 0.5) * brick_height
            R = matmul3(rotz(b.rot), flip_upside())  # 先平面旋转，再绕X翻转
            a,bm,c, d,e,g, h,i,j = R
            f.write(f"1 {b.color} {X:.3f} {Y:.3f} {Z:.3f} {a} {bm} {c} {d} {e} {g} {h} {i} {j} {part}\n")

# ========== 主流程 ==========
def process_one(path, args, out_dirs):
    stem = os.path.splitext(os.path.basename(path))[0]
    png  = os.path.join(out_dirs["plots"],  f"{stem}.png")
    jso  = os.path.join(out_dirs["json"],   f"{stem}.json")
    ldr  = os.path.join(out_dirs["ldr"],    f"{stem}.ldr")
    ldr1 = os.path.join(out_dirs["ldr_1x1"],f"{stem}_1x1.ldr")

    coords = load_voxels(path, args.input_format)
    coords = transform_coords(coords)
    if args.make_plots:
        visualize_coords(coords, png, title=stem, show=args.show)

    V = coords_to_grid(coords)

    # 先 1×1 完整覆盖（绝不丢体素）
    bricks_1x1 = one_by_one_bricks(V)
    if args.export_1x1_ldr:
        write_ldr(bricks_1x1, ldr1, invert_z=args.invert_z)

    # 合并（不跨层）
    merged = merge_all_layers_from_grid(V)

    # 覆盖校验；不一致则回退到 1×1
    voxels_num = V.sum()
    cover_num = sum(b.dx*b.dy for b in merged)
    if cover_num != voxels_num:
        print(f"[WARN] {stem}: merged cover={cover_num} != voxels={voxels_num}, fallback to 1x1")
        merged = bricks_1x1

    write_json(merged, jso)
    write_ldr(merged, ldr, invert_z=args.invert_z)
    print(f"[OK] {stem}: vox={voxels_num} bricks={len(merged)} -> {jso} | {ldr}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",  required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--input-format", choices=["npy","coords"], default="npy")
    ap.add_argument("--pattern", default=".npy")
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--export-1x1-ldr", action="store_true", help="另存纯 1×1 LDR 用于对比")
    ap.add_argument("--invert-z", action="store_true", help="将 LDraw 的 Z 取反（可选）")
    args = ap.parse_args()

    out_dirs = {
        "json":    os.path.join(args.output_dir, "json"),
        "ldr":     os.path.join(args.output_dir, "ldr"),
        "ldr_1x1": os.path.join(args.output_dir, "ldr_1x1"),
        "plots":   os.path.join(args.output_dir, "plots"),
    }
    for d in out_dirs.values(): ensure_dir(d)

    files = [os.path.join(args.input_dir,f) for f in os.listdir(args.input_dir) if f.endswith(args.pattern)]
    files.sort()
    if not files:
        print(f"[WARN] no files in {args.input_dir} with pattern '{args.pattern}'"); return

    print(f"[INFO] found {len(files)} files")
    for p in files:
        try:
            process_one(p, args, out_dirs)
        except Exception as e:
            print(f"[ERR] {os.path.basename(p)}: {e}")
    print("[DONE]")

if __name__ == "__main__":
    main()