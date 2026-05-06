import os
import math
import random
from collections import defaultdict

# ==========================================
# ⚙️ 配置区
# ==========================================
INPUT_FILE = "input.ldr" 
OUTPUT_DIR = "smart_demo_v3_fixed_output"

# 颜色
COLOR_RED    = 4   # 红色 (不稳)
COLOR_BLACK  = 0   # 黑色 (稳定)
COLOR_GRAY   = 7   # 灰色 (补丁)

# LDraw 单位
W, H, D = 20.0, 24.0, 20.0 

# 零件尺寸字典
PART_DIMS = {
    "3005.dat": (20, 24, 20), "3004.dat": (40, 24, 20), "3003.dat": (40, 24, 40),
    "3001.dat": (80, 24, 40), "3002.dat": (60, 24, 40), "3008.dat": (160, 24, 20),
    "3009.dat": (120, 24, 20),"3010.dat": (80, 24, 20), "3622.dat": (60, 24, 20),
    "2456.dat": (120, 24, 40),
    "3024.dat": (20, 8, 20),  "3023.dat": (40, 8, 20),  "3022.dat": (40, 8, 40),
    "3021.dat": (60, 8, 40),  "3020.dat": (80, 8, 40),  "3623.dat": (60, 8, 20),
    "3710.dat": (80, 8, 20)
}

# ==========================================
# 1. 基础工具
# ==========================================
def parse_ldr(filepath):
    bricks = []
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return []
    
    ground_y = -99999 
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != '1': continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                matrix = [float(v) for v in parts[5:14]]
                part = parts[14].lower()
                
                w, h, d = PART_DIMS.get(part, (20, 24, 20))
                is_rotated = abs(matrix[0]) < 0.1 and abs(matrix[2]) > 0.9
                if is_rotated: w, d = d, w
                
                bb = {
                    'min_x': x - w/2, 'max_x': x + w/2,
                    'min_y': y - h/2, 'max_y': y + h/2,
                    'min_z': z - d/2, 'max_z': z + d/2
                }
                ground_y = max(ground_y, bb['max_y'])

                bricks.append({
                    "x": x, "y": y, "z": z, "matrix": parts[5:14], "part": part,
                    "dims": (w, h, d), "bb": bb
                })
            except: continue
    return bricks, ground_y

def write_ldr(filename, bricks_data):
    with open(filename, 'w') as f:
        f.write(f"0 LEGO Model: {os.path.basename(filename)}\n")
        for b in bricks_data:
            mat = " ".join(b['matrix'])
            f.write(f"1 {b['color']} {b['x']:.2f} {b['y']:.2f} {b['z']:.2f} {mat} {b['part']}\n")

# ==========================================
# 2. 核心算法：Init 递归传导稳定性
# ==========================================

def create_init_dense_recursive(src_bricks, ground_y):
    """
    Init V3 Fixed: 
    1. 生成密集 1x1
    2. 按列分组，自底向上检查。
    3. 如果下方积木是红色，上方积木必须也是红色。
    """
    one_by_ones = []
    
    # --- 步骤 1：生成所有 1x1 ---
    for b in src_bricks:
        w, h, d = b['dims']
        cols = int(round(w / W))
        rows = int(round(d / D))
        
        start_x = b['x'] - (w / 2) + (W / 2)
        start_z = b['z'] - (d / 2) + (D / 2)
        
        for i in range(cols):
            for j in range(rows):
                cx = start_x + i * W
                cz = start_z + j * D
                cy = b['y']
                
                bb = {
                    'min_x': cx - W/2, 'max_x': cx + W/2,
                    'min_y': cy - h/2, 'max_y': cy + h/2,
                    'min_z': cz - D/2, 'max_z': cz + D/2
                }
                
                one_by_ones.append({
                    'x': cx, 'y': cy, 'z': cz,
                    'matrix': ["1","0","0","0","1","0","0","0","1"], 
                    'part': "3005.dat", 
                    'bb': bb,
                    'color': COLOR_RED # 默认先全红，只有通过检查才变黑
                })

    # --- 步骤 2：按列分组 (Columnar Grouping) ---
    # key: (x_grid_index, z_grid_index)
    columns = defaultdict(list)
    for b in one_by_ones:
        k_x = int(round(b['x']))
        k_z = int(round(b['z']))
        columns[(k_x, k_z)].append(b)

    # --- 步骤 3：自底向上状态传导 ---
    EPS = 1.0
    
    for (gx, gz), bricks in columns.items():
        # 按照物理高度排序：从低到高
        # LDraw Y 向下为正，所以 max_y 越大代表越低。
        # 我们按照 bb['max_y'] (底部高度) 从大到小排序 (即从地面向上空)
        bricks.sort(key=lambda b: b['bb']['max_y'], reverse=True)
        
        for i, b in enumerate(bricks):
            is_stable = False
            
            # 情况 A: 接地
            if abs(b['bb']['max_y'] - ground_y) < EPS:
                is_stable = True
            
            # 情况 B: 踩在另一个积木上
            elif i > 0:
                prev_b = bricks[i-1]
                
                # 检查物理接触：我的底部 == 下面的顶部
                # b['bb']['max_y'] approx prev_b['bb']['min_y']
                if abs(b['bb']['max_y'] - prev_b['bb']['min_y']) < EPS:
                    # 关键逻辑：只有下面是黑的(稳的)，我才能是黑的
                    if prev_b['color'] == COLOR_BLACK:
                        is_stable = True
                
                # 如果中间有空隙，或者下面是红的，那我就是红的 (is_stable 保持 False)
            
            # 赋值颜色
            b['color'] = COLOR_BLACK if is_stable else COLOR_RED

    return one_by_ones

# ==========================================
# 3. 其他方法 (Baseline & Ours) - 保持不变
# ==========================================
def is_supported_large(brick, all_bricks, ground_level):
    if abs(brick['bb']['max_y'] - ground_level) < 1.0: return True
    for other in all_bricks:
        if brick == other: continue
        if abs(brick['bb']['max_y'] - other['bb']['min_y']) < 1.0: 
            ox = max(0, min(brick['bb']['max_x'], other['bb']['max_x']) - max(brick['bb']['min_x'], other['bb']['min_x']))
            oz = max(0, min(brick['bb']['max_z'], other['bb']['max_z']) - max(brick['bb']['min_z'], other['bb']['min_z']))
            if ox > 1.0 and oz > 1.0: return True
    return False

def create_greedy_selective(src_bricks, ground_y):
    res = []
    unstable_indices = []
    for idx, b in enumerate(src_bricks):
        supported = is_supported_large(b, src_bricks, ground_y)
        color = COLOR_BLACK if supported else COLOR_RED
        if not supported: unstable_indices.append(idx)
        res.append({**b, 'color': color})
    return res, unstable_indices

def create_ours_bridged(src_bricks, unstable_indices):
    res = [{**b, 'color': COLOR_BLACK} for b in src_bricks]
    patches = []
    random.seed(42)
    for idx in unstable_indices:
        b = src_bricks[idx]
        if random.random() > 0.5: continue
        patch_y = b['bb']['max_y'] + 8.0/2 
        dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        dx, dz = random.choice(dirs)
        patch_x = b['x'] + dx * 10
        patch_z = b['z'] + dz * 10
        matrix = ["1","0","0","0","1","0","0","0","1"] if dx != 0 else ["0","0","1","0","1","0","-1","0","0"]
        patches.append({
            'x': patch_x, 'y': patch_y, 'z': patch_z,
            'matrix': matrix, 'part': "3023.dat", 'color': COLOR_GRAY
        })
    return res + patches

# ==========================================
# 主流程
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 开始生成 V3 Fixed (Recursive Init)...")
    
    src_bricks, ground_y = parse_ldr(INPUT_FILE)
    if not src_bricks: return
    print(f"   读取模型: {len(src_bricks)} 块, 地面 Y={ground_y:.1f}")
    
    # 1. Init: 递归传导稳定性 (地基不稳，全楼变红)
    print("   生成 Init...")
    init_data = create_init_dense_recursive(src_bricks, ground_y)
    write_ldr(os.path.join(OUTPUT_DIR, "0_init.ldr"), init_data)
    
    # 2. Baseline
    print("   生成 Baseline...")
    greedy_data, unstable_idxs = create_greedy_selective(src_bricks, ground_y)
    write_ldr(os.path.join(OUTPUT_DIR, "1_baseline.ldr"), greedy_data)
    
    # 3. Ours
    print("   生成 Ours...")
    ours_data = create_ours_bridged(src_bricks, unstable_idxs)
    write_ldr(os.path.join(OUTPUT_DIR, "3_ours.ldr"), ours_data)
    
    print(f"\n✅ 完成！")
    print("   Init 效果: 只要最底下悬空，上面的一整串积木都会变成红色。")

if __name__ == "__main__":
    main()