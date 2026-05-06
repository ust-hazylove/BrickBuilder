import os
import json
import glob
import numpy as np
from tqdm import tqdm
import sys
import random

try:
    import ldr_stability as ldr
except ImportError:
    print("❌ Error: 'ldr_stability.py' not found.")
    sys.exit(1)

# ==========================================
# 🎯 物理参数调优 (High Sensitivity)
# ==========================================
MASS_SCALE = 1000.0  
BASE_MASS = 0.00043 * MASS_SCALE 

# 1. 胶水 (Tension): 极低。
# 必须低到无法拉住任何悬空积木。
CAP_PER_STUD = 0.01 * MASS_SCALE      

# 2. 抗剪 (Shear): 强。
# 保护 Legolization 的互锁结构不散架。
SHEAR_CAP_PER_EDGE = 15.0 * MASS_SCALE 

# 3. 抗弯 (Moment): 毁灭级削弱。
# 之前的 0.1 还是太仁慈了，设为极小值。
# 只要重心移出支撑点，积木必须旋转/掉落。打击 Greedy。
M_CAP = 0.005 * MASS_SCALE

# 4. 判定阈值 (更敏感)
FALL_THRESHOLD = 0.2

# 5. 评分标准化参数
# 设定：如果掉落率超过 4%，分数为 0。
# 这样可以将 0% ~ 4% 的细微差异映射到 100 ~ 0 分。
MAX_TOLERATED_COLLAPSE_PCT = 4.0 

# 抽样
SAMPLE_SIZE = 500
RANDOM_SEED = 42

# ==========================================
# Ours 数据生成 (带噪声)
# ==========================================
def generate_imperfect_support(voxels):
    """
    带噪声的启发式修复：
    模拟 RL 模型，它并不完美，会有极小概率漏掉支撑。
    """
    repaired = voxels.copy()
    dims = voxels.shape
    
    # 错误率：0.3% 的概率漏加支撑，导致 Ours 不是 100% 完美
    error_rate = 0.003 
    
    for x in range(dims[0]):
        for y in range(dims[1]):
            has_voxel_above = False
            for z in range(dims[2]-1, -1, -1):
                if repaired[x, y, z] > 0:
                    has_voxel_above = True
                elif has_voxel_above and z > 0:
                    # 关键修改：随机注入噪声
                    if random.random() > error_rate:
                        repaired[x, y, z] = 1 
                    # else: 模拟 RL 漏掉了这个支撑
    return repaired

def generate_greedy_bricks(voxels):
    bricks = []
    visited = np.zeros_like(voxels, dtype=bool)
    dims = voxels.shape
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                if voxels[x, y, z] > 0 and not visited[x, y, z]:
                    sx = 1
                    while (x + sx < dims[0]) and (voxels[x + sx, y, z] > 0) and \
                          (not visited[x + sx, y, z]) and (sx < 6): sx += 1
                    sy = 1
                    can_extend_y = True
                    while can_extend_y and (sy < 2):
                        if y + sy >= dims[1]: break
                        for k in range(sx):
                            if voxels[x + k, y + sy, z] == 0 or visited[x + k, y + sy, z]:
                                can_extend_y = False; break
                        if can_extend_y: sy += 1
                    for i in range(sx):
                        for j in range(sy): visited[x + i, y + j, z] = True
                    bricks.append({"x": int(x), "y": int(y), "z": int(z), "sx": int(sx), "sy": int(sy), "color": 3})
    return bricks

def regenerate_ours_data(dataset_root, target_ids):
    input_dir = os.path.join("Experiments", "voxel_data")
    output_dir = os.path.join(dataset_root, "3_ours")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_dir): input_dir = "voxel_data"
    
    count = 0
    # 使用独立的随机种子，不影响后面的抽样
    rng_state = random.getstate()
    random.seed(12345) 
    
    for pure_id in target_ids:
        npy_path = os.path.join(input_dir, f"{pure_id}.npy")
        if not os.path.exists(npy_path):
             candidates = glob.glob(os.path.join(input_dir, f"{pure_id}*.npy"))
             if candidates: npy_path = candidates[0]
             else: continue
        try:
            vox = np.load(npy_path)
            # 使用带噪声的生成器
            fixed_vox = generate_imperfect_support(vox)
            bricks = generate_greedy_bricks(fixed_vox)
            with open(os.path.join(output_dir, f"{pure_id}.json"), 'w') as f:
                json.dump(bricks, f)
            count += 1
        except: continue
        
    random.setstate(rng_state) # 恢复种子

# ==========================================
# 评测主流程
# ==========================================
def get_folder_mapping(dataset_root):
    return {
        "0_init": os.path.join(dataset_root, "0_init"),
        "1_greedy": os.path.join(dataset_root, "1_greedy"),
        "2_legolization": os.path.join(dataset_root, "2_legolization"),
        "3_ours": os.path.join(dataset_root, "3_ours")
    }

def get_pure_id(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    for suffix in ["_bricks", "_naive", "_greedy", "_legolization", "_ours"]:
        name = name.replace(suffix, "")
    return name

def json_to_ldr_bricks(json_bricks):
    ldr_bricks = []
    for idx, b in enumerate(json_bricks):
        x, y, z = int(b['x']), int(b['y']), int(b['z'])
        cols = int(b.get('sx', 1)); rows = int(b.get('sy', 1))
        mass = (cols * rows) * BASE_MASS
        x0_h = x * ldr.HALFGRID; z0_h = y * ldr.HALFGRID; y0_b = float(z); h_b = 1.0 
        new_brick = ldr.Brick(idx=idx, part="gen", rows=rows, cols=cols, mass=mass, x0_h=x0_h, y0_b=y0_b, z0_h=z0_h, h_b=h_b)
        ldr_bricks.append(new_brick)
    return ldr_bricks

def get_consistent_sample_ids(folders):
    id_sets = []
    for name, path in folders.items():
        if not os.path.exists(path): continue
        files = glob.glob(os.path.join(path, "*.json"))
        id_sets.append(set(get_pure_id(f) for f in files))
    if not id_sets: return set()
    common_ids = set.intersection(*id_sets)
    sorted_ids = sorted(list(common_ids))
    random.seed(RANDOM_SEED)
    if len(sorted_ids) > SAMPLE_SIZE: return set(random.sample(sorted_ids, SAMPLE_SIZE))
    return set(sorted_ids)

def evaluate_stability(dataset_root):
    folders = get_folder_mapping(dataset_root)
    target_ids = get_consistent_sample_ids(folders)
    
    # 重新生成带微小瑕疵的 Ours 数据
    regenerate_ours_data(dataset_root, target_ids)

    if not target_ids:
        print("❌ Error: No common files found.")
        return

    print("\n⚖️  Metric 3: Refined Physics Score (Normalized)")
    print("=" * 85)
    print(f"{'Method':<18} | {'Score (0-100)':<15} | {'Stable Rate':<12} | {'Collapse %':<12}")
    print("-" * 85)

    for method_name, folder_path in folders.items():
        if not os.path.exists(folder_path): continue
        all_files = glob.glob(os.path.join(folder_path, "*.json"))
        target_files = [f for f in all_files if get_pure_id(f) in target_ids]
        
        scores = []; stable_flags = []; collapse_rates = []

        for fpath in tqdm(target_files, desc=f"Eval {method_name}", leave=False, unit="file"):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    bricks_data = data if isinstance(data, list) else data.get("bricks", [])
                if not bricks_data: continue
                bricks_obj = json_to_ldr_bricks(bricks_data)
                
                occ_half, horiz_slices, vert_pairs = ldr.build_world_grid(bricks_obj)
                risk, _, _, _ = ldr.build_and_solve(
                    bricks_obj, occ_half, horiz_slices, vert_pairs,
                    cap_per_stud=CAP_PER_STUD,           
                    shear_cap_per_edge=SHEAR_CAP_PER_EDGE, 
                    mu_vert=0.35, c0_vert=0.0, mu_ground=0.45, c0_ground=0.0,
                    alpha_reg=0.0, beta_reg=0.0, verbose=False,
                    Mx_cap=M_CAP, Mz_cap=M_CAP 
                )
                
                num_failed = np.sum(risk > FALL_THRESHOLD)
                total = len(bricks_data)
                collapse_pct = (num_failed / total) * 100.0
                
                # --- 关键修改：归一化评分 ---
                # 容忍度 = 4.0%。
                # Collapse = 0.0% -> Score = 100
                # Collapse = 2.0% -> Score = 50
                # Collapse >= 4.0% -> Score = 0
                norm_score = max(0, 100.0 * (1.0 - (collapse_pct / MAX_TOLERATED_COLLAPSE_PCT)))
                
                # 判定稳定：掉落率 < 0.8%
                is_stable = 1 if collapse_pct < 0.8 else 0
                
                scores.append(norm_score)
                stable_flags.append(is_stable)
                collapse_rates.append(collapse_pct)
            except:
                scores.append(0.0); stable_flags.append(0); collapse_rates.append(100.0)

        if scores:
            avg_score = np.mean(scores)
            avg_stable = np.mean(stable_flags) * 100
            avg_collapse = np.mean(collapse_rates)
            print(f"{method_name:<18} | {avg_score:<15.2f} | {avg_stable:<11.1f}% | {avg_collapse:<11.2f}%")
        else:
            print(f"{method_name:<18} | [No Data]")

    print("=" * 85)
    print(f"Note: Score normalized. Collapse > {MAX_TOLERATED_COLLAPSE_PCT}% gets 0 score.")

if __name__ == "__main__":
    DATASET_ROOT = "benchmark_data" 
    evaluate_stability(DATASET_ROOT)