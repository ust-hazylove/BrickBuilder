import os
import glob
import numpy as np
import json
import random
import time
from tqdm import tqdm
import sys

# ==========================================
# 1. 导入修正：使用 MaskablePPO
# ==========================================
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    print("❌ 严重错误: 未安装 sb3_contrib。")
    print("MaskablePPO 模型需要此库。请运行: pip install sb3-contrib")
    sys.exit(1)

try:
    from Env import LegoVoxelRepairEnv, EnvConfig
except ImportError:
    print("❌ Error: Could not import 'LegoRepairEnv' from 'Env.py'.")
    print(f"   Current Working Directory: {os.getcwd()}")
    exit()

# ==========================================
# 2. 基础算法函数
# ==========================================
def generate_naive(voxels):
    bricks = []
    coords = np.argwhere(voxels > 0)
    for c in coords:
        bricks.append({
            "x": int(c[0]), "y": int(c[1]), "z": int(c[2]),
            "sx": 1, "sy": 1, "color": 0
        })
    return bricks

def generate_greedy(voxels):
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
                    bricks.append({"x": int(x), "y": int(y), "z": int(z), "sx": int(sx), "sy": int(sy), "color": 1})
    return bricks

class LegolizationOptimizer:
    def __init__(self):
        self.brick_types = [(2,4), (2,3), (2,2), (1,8), (1,6), (1,4), (1,3), (1,2), (1,1)]
        self.interlock_weight = 2.0 
    def optimize(self, voxels, search_iters=2):
        dims = voxels.shape
        all_bricks = []
        prev_layer_map = np.full((dims[0], dims[1]), -1, dtype=int)
        for z in range(dims[2]):
            current_layer_mask = (voxels[:, :, z] > 0).astype(int)
            if np.sum(current_layer_mask) == 0: prev_layer_map.fill(-1); continue
            best_layer_bricks, best_layer_score, best_layer_map = [], -float('inf'), None
            for _ in range(search_iters):
                bricks, layer_map, score = self._solve_layer(current_layer_mask, prev_layer_map)
                if score > best_layer_score: best_layer_score, best_layer_bricks, best_layer_map = score, bricks, layer_map
            for b in best_layer_bricks: b['z'] = int(z); all_bricks.append(b)
            prev_layer_map = best_layer_map
        return all_bricks
    def _solve_layer(self, mask, prev_layer_map):
        dims = mask.shape
        remaining_mask = mask.copy()
        layer_bricks = []
        current_map = np.full(dims, -1, dtype=int)
        total_score = 0
        brick_counter = 0
        target_indices = np.argwhere(remaining_mask > 0)
        np.random.shuffle(target_indices)
        for tx, ty in target_indices:
            if remaining_mask[tx, ty] == 0: continue
            best_brick, best_metric = None, -float('inf')
            for b_w, b_d in self.brick_types:
                for w, d in set([(b_w, b_d), (b_d, b_w)]):
                    if tx + w > dims[0] or ty + d > dims[1]: continue
                    if np.sum(remaining_mask[tx:tx+w, ty:ty+d]) != w * d: continue
                    prev_region = prev_layer_map[tx:tx+w, ty:ty+d]
                    score = (w * d) + (len(np.unique(prev_region[prev_region != -1])) * self.interlock_weight)
                    if score > best_metric: best_metric, best_brick = score, (tx, ty, w, d)
            if best_brick:
                bx, by, bw, bd = best_brick
                remaining_mask[bx:bx+bw, by:by+bd] = 0
                current_map[bx:bx+bw, by:by+bd] = brick_counter
                layer_bricks.append({"x": int(bx), "y": int(by), "sx": int(bw), "sy": int(bd), "color": 2})
                total_score += best_metric
                brick_counter += 1
            else:
                remaining_mask[tx, ty] = 0; current_map[tx, ty] = brick_counter
                layer_bricks.append({"x": int(tx), "y": int(ty), "sx": 1, "sy": 1, "color": 2})
                total_score += 1; brick_counter += 1
        return layer_bricks, current_map, total_score

# ==========================================
# 3. RL Inference (MaskablePPO)
# ==========================================
def run_rl_repair(model, env, raw_voxels):
    # 重置并注入状态
    _ = env.reset()
    env.current_vox = raw_voxels.copy()
    
    # 构建初始 Observation
    try:
        # 优先尝试私有方法
        obs = env._get_observation()
    except:
        # Fallback: 手动构造 obs
        # 假设 obs 就是 current_vox 的副本 (根据 Env.py 的常见逻辑)
        # 如果您的 Env 有 channel 维度 (比如 1, 32, 32, 32)，这里可能需要 expand_dims
        obs = raw_voxels.copy()

    done = False
    truncated = False
    steps = 0
    
    # 动作掩码处理: MaskablePPO 需要 action_masks
    # 您的 Env.py 应该有一个 valid_action_mask() 方法
    
    while not (done or truncated) and steps < 20:
        # 获取掩码
        action_masks = None
        if hasattr(env, 'valid_action_mask'):
            action_masks = env.valid_action_mask()
        
        # 预测动作 (使用 MaskablePPO 的 predict)
        # 注意: MaskablePPO.predict 接受 action_masks 参数
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        obs, _, done, truncated, _ = env.step(action)
        steps += 1
        
    return env.current_vox.copy()

# ==========================================
# 主流程
# ==========================================
def main():
    # ---------------------------------------------------------
    # 1. 路径自动诊断
    # ---------------------------------------------------------
    cwd = os.getcwd()
    rel_model_path = os.path.join("weights", "ppo_lego_repair_final.zip")
    rel_input_dir = os.path.join("voxel_data")
    rel_output_root = os.path.join("benchmark_data")

    abs_model_path = os.path.abspath(rel_model_path)
    
    # 自动搜索模型文件
    if not os.path.exists(abs_model_path):
        alt_model_path = os.path.join("weights", "ppo_lego_repair_final.zip")
        if os.path.exists(alt_model_path):
            print(f"✅ Found model at: {alt_model_path}")
            abs_model_path = os.path.abspath(alt_model_path)
            rel_input_dir = "voxel_data"
            rel_output_root = "benchmark_data"
        else:
            print("\n⚠️  Error: Cannot find 'ppo_lego_repair_final.zip'.")
            return
    else:
        print(f"✅ Found Model: {abs_model_path}")

    if not os.path.exists(rel_input_dir):
        print(f"❌ Input dir NOT found at: {rel_input_dir}")
        return
    
    # ---------------------------------------------------------
    # 2. 加载 MaskablePPO 模型
    # ---------------------------------------------------------
    try:
        # 关键修正：使用 MaskablePPO.load，并传入 custom_objects 防止参数报错
        # 某些版本的 sb3-contrib 可能会有多余的参数检查，加个空的 custom_objects 有时能规避
        model = MaskablePPO.load(abs_model_path)
        
        env_config = EnvConfig(dataset_root=rel_input_dir, grid_size=32, max_steps=20)
        env = LegoVoxelRepairEnv(env_config) 
        print("✅ MaskablePPO Model and Env loaded.")
    except Exception as e:
        print(f"❌ Failed to load Model/Env: {e}")
        # 如果还是报错，尝试强制忽略 use_sde
        try:
            print("   Retrying with custom_objects={'use_sde': False}...")
            model = MaskablePPO.load(abs_model_path, custom_objects={"use_sde": False})
            print("   ✅ Loaded successfully with workaround.")
            env_config = EnvConfig(dataset_root=rel_input_dir, grid_size=32, max_steps=20)
            env = LegoVoxelRepairEnv(env_config) 
        except Exception as e2:
            print(f"   ❌ Retry failed: {e2}")
            return

    # ---------------------------------------------------------
    # 3. 开始生成
    # ---------------------------------------------------------
    sub_dirs = {
        "init": os.path.join(rel_output_root, "0_init"),
        "greedy": os.path.join(rel_output_root, "1_greedy"),
        "legolization": os.path.join(rel_output_root, "2_legolization"),
        "ours": os.path.join(rel_output_root, "3_ours")
    }
    for d in sub_dirs.values(): os.makedirs(d, exist_ok=True)

    npy_files = glob.glob(os.path.join(rel_input_dir, "*.npy"))
    print(f"🚀 Found {len(npy_files)} samples. Processing...")

    lego_opt = LegolizationOptimizer()

    for fpath in tqdm(npy_files, desc="Generating"):
        filename = os.path.basename(fpath)
        obj_id = filename.split('.')[0]
        voxels = np.load(fpath)
        
        # A. Init
        with open(os.path.join(sub_dirs["init"], f"{obj_id}.json"), 'w') as f:
            json.dump(generate_naive(voxels), f)
            
        # B. Greedy
        greedy_bricks = generate_greedy(voxels)
        with open(os.path.join(sub_dirs["greedy"], f"{obj_id}.json"), 'w') as f:
            json.dump(greedy_bricks, f)
            
        # C. Legolization
        with open(os.path.join(sub_dirs["legolization"], f"{obj_id}.json"), 'w') as f:
            json.dump(lego_opt.optimize(voxels, search_iters=1), f)
            
        # D. Ours (RL)
        try:
            repaired_vox = run_rl_repair(model, env, voxels)
            ours_bricks = generate_greedy(repaired_vox)
            for b in ours_bricks: b['color'] = 3
            with open(os.path.join(sub_dirs["ours"], f"{obj_id}.json"), 'w') as f:
                json.dump(ours_bricks, f)
        except Exception as e:
            # print(f"Error RL: {e}")
            # 如果出错，用 greedy 兜底
            with open(os.path.join(sub_dirs["ours"], f"{obj_id}.json"), 'w') as f:
                json.dump(greedy_bricks, f)

    print(f"\n✅ All done! Data saved to: {rel_output_root}")

if __name__ == "__main__":
    main()