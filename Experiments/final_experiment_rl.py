import os
import json
import time
import glob
import numpy as np
import cvxpy as cp
import torch
import pandas as pd
from tqdm import tqdm

# ==========================================
# 0. 导入依赖
# ==========================================
# 尝试导入 StableLego (仅作备用)
try:
    from Experiments.voxel_stability import stability_score
except ImportError:
    try:
        from Experiments.voxel_stability import stability_score
    except ImportError:
        pass

# 导入 RL
try:
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from Env import LegoVoxelRepairEnv, EnvConfig
    from train import Lego3DCNN, mask_fn 
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# ==========================================
# 1. RL Agent
# ==========================================
class RLRepairAgent:
    def __init__(self, weights_path):
        self.model = None
        self.env = None
        if not RL_AVAILABLE: return
        print(f"Loading RL Agent from: {weights_path}")
        if not os.path.exists(weights_path):
            print(f"Weights not found. Using Greedy Simulation.")
            return
        self.env_config = EnvConfig()
        self.env = LegoVoxelRepairEnv(self.env_config)
        self.env = ActionMasker(self.env, mask_fn)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = MaskablePPO.load(weights_path, env=self.env, device=device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def predict(self, input_voxel_grid):
        if self.model is None:
            return self._greedy_merge_postprocess(input_voxel_grid)
        
        self.env.reset()
        if hasattr(self.env.unwrapped, 'current_vox'):
            target_shape = self.env.unwrapped.current_vox.shape
        else:
            target_obs_shape = self.env.observation_space.shape
            target_shape = target_obs_shape[1:] if len(target_obs_shape)==4 else target_obs_shape

        # Inject Data
        current_vox = np.zeros(target_shape, dtype=int)
        D, H, W = input_voxel_grid.shape
        tD, tH, tW = target_shape
        sd, sh, sw = (tD - D) // 2, (tH - H) // 2, (tW - W) // 2
        
        src_d_start, src_d_end = max(0, -sd), min(D, tD - sd)
        src_h_start, src_h_end = max(0, -sh), min(H, tH - sh)
        src_w_start, src_w_end = max(0, -sw), min(W, tW - sw)
        dst_d_start, dst_d_end = max(0, sd), min(tD, sd + D)
        dst_h_start, dst_h_end = max(0, sh), min(tH, sh + H)
        dst_w_start, dst_w_end = max(0, sw), min(tW, sw + W)

        if (dst_d_end > dst_d_start):
            current_vox[dst_d_start:dst_d_end, dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
                (input_voxel_grid[src_d_start:src_d_end, src_h_start:src_h_end, src_w_start:src_w_end] > 0).astype(int)

        self.env.unwrapped.current_vox = current_vox.copy()

        # Construct Obs
        obs_shape = self.env.observation_space.shape
        obs = np.zeros(obs_shape, dtype=np.float32)
        obs[0] = current_vox.astype(np.float32)
        if obs_shape[0] > 1 and hasattr(self.env.unwrapped, 'unstable_mask') and self.env.unwrapped.unstable_mask is not None:
             if self.env.unwrapped.unstable_mask.shape == current_vox.shape:
                 obs[1] = self.env.unwrapped.unstable_mask

        done = False
        steps = 0
        while not done:
            action_masks = self.env.action_masks()
            action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            steps += 1
            if steps > 100: break

        repaired_voxels = self.env.unwrapped.current_vox.copy()
        return self._greedy_merge_postprocess(repaired_voxels)

    def _greedy_merge_postprocess(self, voxels):
        bricks = []
        visited = np.zeros_like(voxels, dtype=bool)
        dims = voxels.shape
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    if voxels[x, y, z] > 0 and not visited[x, y, z]:
                        bricks.append({"x": int(x), "y": int(y), "z": int(z), "sx": 1, "sy": 1, "color": 2})
                        visited[x, y, z] = True
        return bricks

# ==========================================
# 2. Robust Hybrid Solver (The Fix)
# ==========================================
class RobustVoxelSolver:
    """
    双保险求解器：
    1. 尝试 CVXPY 软约束优化 (物理平衡)。
    2. 如果失败或超时，回退到 BFS 连通性计数 (几何连通)。
    保证永远返回有效数值。
    """
    def __init__(self, glue_strength=500.0):
        self.T = glue_strength

    def solve(self, brick_list):
        if not brick_list: return 0.0

        # -----------------------------
        # 策略 A: 几何连通性 (Fallback)
        # -----------------------------
        # 无论如何先算这个，作为底线分数
        connectivity_score = self._calculate_bfs_score(brick_list)
        
        # 如果积木太多(>1000)，直接用 BFS 分数，跳过优化以节省时间
        if len(brick_list) > 1000:
            return connectivity_score

        # -----------------------------
        # 策略 B: 物理优化 (Optimizer)
        # -----------------------------
        try:
            opt_score = self._solve_optimizer(brick_list)
            # 如果优化成功且结果合理，返回优化分数
            if opt_score > -9000:
                return opt_score
        except Exception as e:
            # print(f"Optimizer failed: {e}, using fallback.")
            pass
        
        # 如果优化失败，返回 BFS 分数
        return connectivity_score

    def _calculate_bfs_score(self, brick_list):
        """
        计算有多少积木未连接到地面。
        Score = -1.0 * (悬空积木数量)
        """
        grid = {}
        for idx, b in enumerate(brick_list):
            # 假设 1x1
            grid[(b['x'], b['y'], b['z'])] = idx
            
        # 找地基
        queue = []
        visited = set()
        for pos, idx in grid.items():
            if pos[2] == 0:
                visited.add(idx)
                queue.append(idx)
        
        # BFS
        head = 0
        while head < len(queue):
            curr_idx = queue[head]; head += 1
            b = brick_list[curr_idx]
            x, y, z = b['x'], b['y'], b['z']
            
            # 6 邻域
            neighbors = [(x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z), (x,y,z+1), (x,y,z-1)]
            for np_pos in neighbors:
                if np_pos in grid:
                    n_idx = grid[np_pos]
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)
        
        unstable_count = len(brick_list) - len(visited)
        # 稍微加权一点重力惩罚 (每悬空一个扣 10 分)
        return -10.0 * unstable_count

    def _solve_optimizer(self, brick_list):
        # 简化的软约束优化
        grid = {}
        for idx, b in enumerate(brick_list):
            grid[(b['x'], b['y'], b['z'])] = idx
            
        fixed_indices = {idx for idx, b in enumerate(brick_list) if b['z'] == 0}
        if not fixed_indices: return -5000.0

        edges = []
        for idx, b in enumerate(brick_list):
            x,y,z = b['x'], b['y'], b['z']
            for nx, ny, nz in [(x+1,y,z), (x,y+1,z), (x,y,z+1)]:
                if (nx,ny,nz) in grid:
                    edges.append((idx, grid[(nx,ny,nz)]))

        if not edges: return -100.0 * (len(brick_list) - len(fixed_indices))

        num_bricks = len(brick_list)
        f_glue = cp.Variable(len(edges))
        slacks = cp.Variable(num_bricks)
        
        constrs = [f_glue <= self.T, f_glue >= -self.T]
        
        # 预计算邻接表以加速构建
        adj = [[] for _ in range(num_bricks)]
        for e_i, (u, v) in enumerate(edges):
            adj[u].append((e_i, 1.0))  # Out
            adj[v].append((e_i, -1.0)) # In

        for i in range(num_bricks):
            if i in fixed_indices:
                constrs.append(slacks[i] == 0)
            else:
                # Force Balance: Sum(Glues) + Slack = Gravity(10)
                # 使用 cvxpy 的 sum 表达式
                if not adj[i]:
                    constrs.append(slacks[i] == 10.0)
                else:
                    expr = 0
                    for e_idx, sign in adj[i]:
                        expr += sign * f_glue[e_idx]
                    constrs.append(expr + slacks[i] == 10.0)

        prob = cp.Problem(cp.Minimize(cp.norm(slacks, 1)), constrs)
        
        # 尝试使用更稳健的求解器
        # 如果没有安装 SCS，cvxpy 会自动尝试其他
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except:
            prob.solve(solver=cp.ECOS, verbose=False)
            
        if prob.status in ['optimal', 'optimal_inaccurate']:
            return -prob.value
        else:
            return -9999.0 # Signal failure to fallback

# ==========================================
# 3. Helpers
# ==========================================
def npy_to_naive_bricks(npy_path):
    voxels = np.load(npy_path)
    return [{"x":int(c[0]),"y":int(c[1]),"z":int(c[2]),"sx":1,"sy":1} for c in np.argwhere(voxels>0)]

def ground_structure(brick_list):
    if not brick_list: return []
    min_z = min([b['z'] for b in brick_list])
    return [{**b, 'z': b['z'] - min_z} for b in brick_list]

def calculate_iou(list_a, list_b):
    def to_set(blist):
        s = set()
        for b in blist:
            s.add((b['x'], b['y'], b['z']))
        return s
    sa, sb = to_set(list_a), to_set(list_b)
    inter, union = len(sa & sb), len(sa | sb)
    return (inter/union * 100) if union else 0.0

# ==========================================
# 4. Main
# ==========================================
def run_experiment(data_dir, weights_path):
    input_files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not input_files: print("No files"); return

    agent = RLRepairAgent(weights_path)
    # 使用终极鲁棒求解器
    solver = RobustVoxelSolver(glue_strength=500.0)
    
    records = []
    print(f"\nProcessing {len(input_files)} objects (Robust Mode v9)...")
    print(f"{'ID':<20} | {'Score Before':<12} | {'Score After':<12} | {'IoU':<6}")
    print("-" * 65)

    for fpath in input_files:
        obj_id = os.path.basename(fpath).split('.')[0]
        bricks_before = npy_to_naive_bricks(fpath)
        try:
            raw = np.load(fpath)
            t0 = time.time()
            bricks_after = agent.predict(raw)
            dt = time.time() - t0
        except Exception as e: 
            print(f"Error {obj_id}: {e}")
            continue

        # 1. 强制落地
        b_ground = ground_structure(bricks_before)
        a_ground = ground_structure(bricks_after)
        
        # 2. 求解
        s_b = solver.solve(b_ground)
        s_a = solver.solve(a_ground)
        
        iou = calculate_iou(b_ground, a_ground)

        print(f"{obj_id:<20} | {s_b:<12.1f} | {s_a:<12.1f} | {iou:<6.1f}")

        records.append({
            "Object_ID": obj_id,
            "Score_Before": s_b, 
            "Score_After": s_a,
            "Stability_Gain": s_a - s_b,
            "IoU": iou, 
            "Time": dt
        })

    if records:
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(data_dir, "ablation_results_final.csv"), index=False)
        print(f"\nResults saved to ablation_results_final.csv")

if __name__ == "__main__":
    DATA_DIR = "./voxel_data"
    WEIGHTS_FILE = os.path.join("weights", "ppo_lego_repair_final.zip")
    run_experiment(DATA_DIR, WEIGHTS_FILE)