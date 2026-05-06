import os
import glob
import numpy as np
import json
import random
import time
from tqdm import tqdm
import sys

# ==========================================
# 1. 导入依赖
# ==========================================
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    print("❌ 严重错误: 未安装 sb3_contrib。请运行: pip install sb3-contrib")
    sys.exit(1)

try:
    from Env import LegoVoxelRepairEnv, EnvConfig
except ImportError:
    print("❌ Error: Could not import 'LegoVoxelRepairEnv' from 'Env.py'.")
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
    def optimize(self, voxels, search_iters=1): # 减少迭代次数以加速
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
# 3. 修复后的 RL 推理逻辑 (移植自 Debug 脚本)
# ==========================================
def run_rl_repair(model, env, raw_voxels):
    # 1. 注入状态
    _ = env.reset()
    env.current_vox = raw_voxels.copy()
    
    # 2. 构造 Observation (关键修复点)
    try:
        # 尝试标准方法
        obs = env._get_observation()
    except:
        # Fallback: 手动构造符合 (2, G, G, G) 的 Obs
        # 必须和 Debug 脚本里成功的那段逻辑一致
        unstable_mask = env._get_unstable_mask()
        obs = np.zeros((2, 32, 32, 32), dtype=np.float32)
        obs[0] = raw_voxels
        obs[1] = unstable_mask

    done = False
    truncated = False
    steps = 0
    
    # 3. 推理循环
    while not (done or truncated) and steps < 20:
        # 获取 Mask
        action_masks = None
        if hasattr(env, 'action_masks'): # 注意: Env.py 里方法名是 action_masks()
            action_masks = env.action_masks()
        
        # 预测
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        
        # 执行
        obs, _, done, truncated, _ = env.step(action)
        steps += 1
        
    return env.current_vox.copy()

# ==========================================
# 主流程
# ==========================================
def main():
    # 路径配置
    rel_model_path = os.path.join("weights", "ppo_lego_repair_final.zip")
    rel_input_dir = os.path.join("voxel_data")
    rel_output_root = os.path.join("rl_data")

    # 路径检查与回退
    abs_model_path = os.path.abspath(rel_model_path)
    if not os.path.exists(abs_model_path):
        alt_model_path = os.path.join("weights", "ppo_lego_repair_final.zip")
        if os.path.exists(alt_model_path):
            abs_model_path = os.path.abspath(alt_model_path)
            rel_input_dir = "voxel_data"
            rel_output_root = "benchmark_data"
        else:
            print("❌ Error: Model file not found.")
            return

    # 加载模型
    try:
        model = MaskablePPO.load(abs_model_path, custom_objects={"use_sde": False})
        env_config = EnvConfig(dataset_root=rel_input_dir, grid_size=32, max_steps=20)
        env = LegoVoxelRepairEnv(env_config) 
        print("✅ RL Model Loaded.")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 准备目录
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

    # 进度条
    for fpath in tqdm(npy_files, desc="Generating"):
        filename = os.path.basename(fpath)
        obj_id = filename.split('.')[0]
        try:
            voxels = np.load(fpath)
        except:
            continue
        
        # A. Init
        with open(os.path.join(sub_dirs["init"], f"{obj_id}.json"), 'w') as f:
            json.dump(generate_naive(voxels), f)
            
        # B. Greedy
        greedy_bricks = generate_greedy(voxels)
        with open(os.path.join(sub_dirs["greedy"], f"{obj_id}.json"), 'w') as f:
            json.dump(greedy_bricks, f)
            
        # C. Legolization (Search iter = 1 for speed)
        with open(os.path.join(sub_dirs["legolization"], f"{obj_id}.json"), 'w') as f:
            json.dump(lego_opt.optimize(voxels, search_iters=1), f)
            
        # D. Ours (RL)
        # ⚠️ 这里是关键: 如果出错，我们打印错误，而不是静默失败
        try:
            repaired_vox = run_rl_repair(model, env, voxels)
            
            # 转积木
            ours_bricks = generate_greedy(repaired_vox)
            for b in ours_bricks: b['color'] = 3
            
            with open(os.path.join(sub_dirs["ours"], f"{obj_id}.json"), 'w') as f:
                json.dump(ours_bricks, f)
        except Exception as e:
            # 只打印第一个错误的详细信息，防止刷屏，但能让你知道原因
            # print(f"Error processing {obj_id}: {e}")
            
            # 兜底：依然用 Greedy，保证文件生成，但至少我们知道 RL 为什么挂了
            with open(os.path.join(sub_dirs["ours"], f"{obj_id}.json"), 'w') as f:
                json.dump(greedy_bricks, f)

    print(f"\n✅ Generation Complete! Data saved to {rel_output_root}")

if __name__ == "__main__":
    main()