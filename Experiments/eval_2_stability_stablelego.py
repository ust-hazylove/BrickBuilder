import os
import json
import glob
import numpy as np
from tqdm import tqdm
import sys

# 尝试导入 ldr_stability
try:
    import ldr_stability as ldr
except ImportError:
    print("❌ Error: 'ldr_stability.py' not found in the current directory.")
    print("Please make sure the file is present.")
    sys.exit(1)

# ==========================================
# 配置与常量
# ==========================================
# 质量估算系数 (kg per stud^2)
MASS_PER_STUD = 0.00043  # 参考 1x1 brick 质量

# 判定阈值：如果某块积木的 risk > THRESHOLD，则认为该模型“不稳定”
STABILITY_THRESHOLD = 0.1 

def get_folder_mapping(dataset_root):
    return {
        "0_init": os.path.join(dataset_root, "0_init"),
        "1_greedy": os.path.join(dataset_root, "1_greedy"),
        "2_legolization": os.path.join(dataset_root, "2_legolization"),
        "3_ours": os.path.join(dataset_root, "3_ours")
    }

# ==========================================
# 核心转换逻辑
# ==========================================
def json_to_ldr_bricks(json_bricks):
    """
    将 JSON 积木列表转换为 ldr_stability.Brick 对象列表
    """
    ldr_bricks = []
    
    for idx, b in enumerate(json_bricks):
        # 读取属性
        x, y, z = int(b['x']), int(b['y']), int(b['z'])
        cols = int(b.get('sx', 1))  # sx -> cols (X轴)
        rows = int(b.get('sy', 1))  # sy -> rows (Z轴/Y轴)
        
        # 质量估算
        mass = (cols * rows) * MASS_PER_STUD
        
        # 坐标转换 (Studio Unit to Half-Grid)
        # ldr_stability 使用 half-grid (0.5 stud) 作为整数坐标
        # x0_h: x * 2 (因为 1 stud = 2 half-grids)
        x0_h = x * ldr.HALFGRID
        z0_h = y * ldr.HALFGRID # JSON y -> Physics Z (depth)
        
        # 高度转换
        # y0_b: 直接使用 z 作为高度层 (brick unit)
        y0_b = float(z)
        h_b = 1.0 # 默认全是标准砖高度
        
        # 构造 Brick 对象
        # dataclass Brick: idx, part, rows, cols, mass, x0_h, y0_b, z0_h, h_b
        new_brick = ldr.Brick(
            idx=idx,
            part="gen_brick", # 虚拟零件名
            rows=rows,
            cols=cols,
            mass=mass,
            x0_h=x0_h,
            y0_b=y0_b,
            z0_h=z0_h,
            h_b=h_b
        )
        ldr_bricks.append(new_brick)
        
    return ldr_bricks

# ==========================================
# 评测主循环
# ==========================================
def evaluate_stability(dataset_root):
    folders = get_folder_mapping(dataset_root)
    
    print("\n⚖️  Metric 3: External Stability Referee (LDR Solver)")
    print("=" * 80)
    print(f"{'Method':<18} | {'Stable Rate':<12} | {'Avg Score':<12} | {'Avg Max Risk':<12}")
    print("-" * 80)

    for method_name, folder_path in folders.items():
        if not os.path.exists(folder_path):
            print(f"{method_name:<18} | [Missing Folder]")
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.json"))
        if not files:
            print(f"{method_name:<18} | [Empty Folder]")
            continue
            
        total_risk_scores = []
        is_stable_list = []
        max_risks = []
        
        # 进度条
        for fpath in tqdm(files, desc=f"Eval {method_name}", leave=False, unit="file"):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    bricks_data = data if isinstance(data, list) else data.get("bricks", [])
                
                # 1. 转换数据
                if not bricks_data:
                    continue
                bricks_obj = json_to_ldr_bricks(bricks_data)
                
                # 2. 构建物理世界 (Contacts)
                # 无需 flip Y，因为我们直接映射 z -> y0_b (upright)
                occ_half, horiz_slices, vert_pairs = ldr.build_world_grid(bricks_obj)
                
                # 3. 求解 (Gurobi)
                # 使用默认参数，屏蔽输出
                risk, _, _, _ = ldr.build_and_solve(
                    bricks_obj, occ_half, horiz_slices, vert_pairs,
                    cap_per_stud=12.0, shear_cap_per_edge=4.0,
                    mu_vert=0.35, c0_vert=0.25, mu_ground=0.45, c0_ground=0.50,
                    alpha_reg=0.0, beta_reg=0.0,
                    verbose=False # 静默模式
                )
                
                # 4. 计算指标
                # Risk 是每个积木的残差力。越接近 0 越稳。
                current_max_risk = np.max(risk)
                current_total_risk = np.sum(risk)
                
                # 判定稳定：最大风险小于阈值
                stable = (current_max_risk < STABILITY_THRESHOLD)
                
                # 计算分数：转化 risk 为 0-100 分数
                # 简单的倒数映射: 100 / (1 + total_risk)
                score = 100.0 / (1.0 + current_total_risk)
                
                is_stable_list.append(1 if stable else 0)
                total_risk_scores.append(score)
                max_risks.append(current_max_risk)
                
            except Exception as e:
                # 求解失败通常意味着极度不稳定或模型错误 -> 记为不稳定
                # print(f"Error: {e}")
                is_stable_list.append(0)
                total_risk_scores.append(0.0)
                max_risks.append(100.0) # Penalty

        # 统计结果
        if is_stable_list:
            avg_stable_rate = np.mean(is_stable_list) * 100
            avg_score = np.mean(total_risk_scores)
            avg_max_risk = np.mean(max_risks)
            
            print(f"{method_name:<18} | {avg_stable_rate:<11.1f}% | {avg_score:<12.2f} | {avg_max_risk:<12.4f}")
        else:
            print(f"{method_name:<18} | [No Valid Data]")

    print("=" * 80)
    print("Note: 'Stable Rate' based on max brick risk < 0.1. Higher Score is better.")

if __name__ == "__main__":
    # 数据集根目录
    DATASET_ROOT = "benchmark_data" 
    evaluate_stability(DATASET_ROOT)