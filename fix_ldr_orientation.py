# fix_ldr_orientation.py
# -*- coding: utf-8 -*-
"""
批量修正 LDR 朝向：
- mode=rotate：绕 x/y/z 轴旋转 180 度
- mode=mirror：关于 XY / YZ / XZ 平面镜像
仅修改 type-1 行: 位置 (x,y,z) 与 3x3 旋转矩阵 (a..i)
用法示例见顶部说明。
"""
import argparse, os, glob

def parse_type1(line: str):
    toks = line.strip().split()
    if len(toks) < 15 or toks[0] != "1":
        return None
    color = toks[1]
    x, y, z = map(float, toks[2:5])
    a,b,c,d,e,f,g,h,i = map(float, toks[5:14])
    part = " ".join(toks[14:])
    return dict(color=color, x=x, y=y, z=z,
                a=a,b=b,c=c,d=d,e=e,f=f,g=g,h=h,i=i, part=part)

def fmt_type1(d):
    return ("1 {color} {x:.6f} {y:.6f} {z:.6f} "
            "{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} "
            "{g:.6f} {h:.6f} {i:.6f} {part}\n").format(**d)

def scan_bbox(lines):
    xmn=xmx=ymn=ymx=zmn=zmx=None
    for ln in lines:
        d = parse_type1(ln)
        if d is None: continue
        x,y,z = d["x"], d["y"], d["z"]
        xmn = x if xmn is None else min(xmn,x); xmx = x if xmx is None else max(xmx,x)
        ymn = y if ymn is None else min(ymn,y); ymx = y if ymx is None else max(ymx,y)
        zmn = z if zmn is None else min(zmn,z); zmx = z if zmx is None else max(zmx,z)
    return xmn,xmx,ymn,ymx,zmn,zmx

def apply_rotate_180(d, axis, px, py, pz):
    # 平移到 pivot
    x,y,z = d["x"]-px, d["y"]-py, d["z"]-pz
    # 位置旋转
    if axis == "x":
        y,z = -y, -z
    elif axis == "y":
        x,z = -x, -z
    elif axis == "z":
        x,y = -x, -y
    else:
        raise ValueError("axis must be x/y/z")
    d["x"], d["y"], d["z"] = x+px, y+py, z+pz
    # 姿态旋转 R' = R_axis(pi) @ R
    a,b,c,d1,e,f1,g,h,i = d["a"],d["b"],d["c"],d["d"],d["e"],d["f"],d["g"],d["h"],d["i"]
    if axis == "x":
        # Rx(pi)=[[1,0,0],[0,-1,0],[0,0,-1]]  -> 左乘 => 取反第2、3行
        d["d"],d["e"],d["f"] = -d1,-e,-f1
        d["g"],d["h"],d["i"] = -g,-h,-i
    elif axis == "y":
        # Ry(pi)=[[-1,0,0],[0,1,0],[0,0,-1]] -> 左乘 => 取反第1、3行
        d["a"],d["b"],d["c"] = -a,-b,-c
        d["g"],d["h"],d["i"] = -g,-h,-i
    else: # z
        # Rz(pi)=[[-1,0,0],[0,-1,0],[0,0,1]] -> 左乘 => 取反第1、2行
        d["a"],d["b"],d["c"] = -a,-b,-c
        d["d"],d["e"],d["f"] = -d1,-e,-f1
    return d

def apply_mirror(d, plane, z0):
    # 关于平面镜像：R' = S @ R, 位置按对应轴取镜像
    # XY 平面：z'=2z0-z, S=diag(1,1,-1) -> 取反第3行
    # YZ 平面：x'=2x0-x, S=diag(-1,1,1) -> 取反第1行
    # XZ 平面：y'=2y0-y, S=diag(1,-1,1) -> 取反第2行
    if plane == "z":  # XY 平面
        d["z"] = 2*z0 - d["z"]
        d["g"],d["h"],d["i"] = -d["g"],-d["h"],-d["i"]
    elif plane == "x":  # YZ 平面
        d["x"] = 2*z0 - d["x"]   # 注意此处沿“x轴所在平面”的常量，用变量名 z0 做通用参数
        d["a"],d["b"],d["c"] = -d["a"],-d["b"],-d["c"]
    elif plane == "y":  # XZ 平面
        d["y"] = 2*z0 - d["y"]
        d["d"],d["e"],d["f"] = -d["d"],-d["e"],-d["f"]
    else:
        raise ValueError("plane must be x/y/z (分别表示 YZ/XZ/XY 三个镜像平面)")
    return d

def process_lines(lines, mode, axis, plane, pivot_mode, px, py, pz):
    xmn,xmx,ymn,ymx,zmn,zmx = scan_bbox(lines)
    if xmn is None:
        return lines
    if pivot_mode == "center":
        cx,cy,cz = (0.5*(xmn+xmx), 0.5*(ymn+ymx), 0.5*(zmn+zmx))
    elif pivot_mode == "value":
        cx,cy,cz = px,py,pz
    elif pivot_mode == "zero":
        cx,cy,cz = 0.0,0.0,0.0
    else:
        raise ValueError("pivot must be center/value/zero")

    out=[]
    if mode == "rotate":
        for ln in lines:
            d = parse_type1(ln)
            if d is None:
                out.append(ln if ln.endswith("\n") else ln+"\n"); continue
            d = apply_rotate_180(d, axis, cx,cy,cz)
            out.append(fmt_type1(d))
        head = f"0 ROTATE_180 axis={axis} pivot=({cx:.6f},{cy:.6f},{cz:.6f})\n"
        out.insert(0, head)
    else:  # mirror
        # 对 mirror，我们用 z0 参数承载对应轴的常量（见上方注释）
        z0 = {"x": cx, "y": cy, "z": cz}[plane]
        for ln in lines:
            d = parse_type1(ln)
            if d is None:
                out.append(ln if ln.endswith("\n") else ln+"\n"); continue
            d = apply_mirror(d, plane, z0)
            out.append(fmt_type1(d))
        head = f"0 MIRROR plane={plane} pivot=({cx:.6f},{cy:.6f},{cz:.6f})\n"
        out.insert(0, head)
    return out

def process_file(path_in, out_dir, mode, axis, plane, pivot_mode, px, py, pz):
    with open(path_in, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    out_lines = process_lines(lines, mode, axis, plane, pivot_mode, px, py, pz)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path_in))[0]
    suffix = f"_{mode}_{axis if mode=='rotate' else plane}"
    out_path = os.path.join(out_dir, name + suffix + ".ldr")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"[OK] {path_in} -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Rotate 180° or Mirror LDR (type-1)")
    ap.add_argument("--input", required=True, help="单个 .ldr 或目录")
    ap.add_argument("--out", default="out_fix", help="输出目录")
    ap.add_argument("--mode", choices=["rotate","mirror"], default="rotate",
                    help="rotate=旋转180°, mirror=镜像")
    ap.add_argument("--axis", choices=["x","y","z"], default="x",
                    help="rotate时选择旋转轴；mirror时忽略")
    ap.add_argument("--plane", choices=["x","y","z"], default="z",
                    help="mirror时选择镜像平面（x=YZ, y=XZ, z=XY）")
    ap.add_argument("--pivot", choices=["center","value","zero"], default="center",
                    help="变换枢轴：模型中心、自定义三元组或世界原点")
    ap.add_argument("--px", type=float, default=0.0, help="pivot=value 时的 x")
    ap.add_argument("--py", type=float, default=0.0, help="pivot=value 时的 y")
    ap.add_argument("--pz", type=float, default=0.0, help="pivot=value 时的 z")
    args = ap.parse_args()

    paths = [args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input, "*.ldr")))
    if not paths:
        print("[ERR] 未找到 .ldr"); return
    for p in paths:
        process_file(p, args.out, args.mode, args.axis, args.plane,
                     args.pivot, args.px, args.py, args.pz)

if __name__ == "__main__":
    main()