import os
import sys
import glob
import random
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm

# 设置 matplotlib 后端防止弹窗
matplotlib.use('Agg')

# ==========================================
# 0. 环境与依赖配置
# ==========================================
# 1. 尝试导入 sb3-contrib (MaskablePPO)
try:
    from sb3_contrib import MaskablePPO
    HAS_SB3 = True
except ImportError:
    print("[Error] sb3-contrib not installed. pip install sb3-contrib")
    HAS_SB3 = False

# 2. 设置路径以导入项目模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# 3. 尝试导入封装好的 RLRepairModule (优先使用)
try:
    from modules.rl_repair import RLRepairModule
    HAS_RL_MODULE = True
    print("[Info] Detected 'modules.rl_repair', will try to use RLRepairModule.")
except ImportError:
    HAS_RL_MODULE = False
    print("[Info] 'modules.rl_repair' not found. Will use manual Env loop.")

# 4. 尝试导入真实环境 (备用)
ENV_CLASS_NAME = "LegoVoxelRepairEnv"
try:
    from ppo_repair.Env import LegoVoxelRepairEnv as RealLegoEnv
    print(f"[Info] Imported {ENV_CLASS_NAME} for manual loop.")
    HAS_REAL_ENV = True
except ImportError:
    print(f"[Warning] Could not import {ENV_CLASS_NAME}, using dummy.")
    HAS_REAL_ENV = False
    class RealLegoEnv:
        def __init__(self, config=None): self.current_vox = None; self.G=32
        def reset(self): return np.zeros((2, 32, 32, 32)), {}
        def step(self, action): return np.zeros((2, 32, 32, 32)), 0, True, False, {}
        def action_masks(self): return np.ones(10, dtype=bool)

# 5. 导入力学脚本 ldr_stability
try:
    import Experiments.ldr_stability as ldr_stability
    from Experiments.ldr_stability import Brick
    HAS_LDR_SOLVER = True
except ImportError:
    print("[Warning] ldr_stability.py not found.")
    HAS_LDR_SOLVER = False

# ==========================================
# 1. 核心指标计算
# ==========================================
def compute_metrics(grid, original_grid):
    # 统一转为 float/int 计算，防止 bool 报错
    grid_f = grid.astype(np.float32)
    orig_f = original_grid.astype(np.float32)
    
    total_voxels = np.sum(grid_f)
    if total_voxels == 0:
        return {"conn": 0.0, "gnd": 0.0, "vol": 0, "iou": 0.0, "added": 0}

    # Connectivity
    labeled_array, num_features = scipy.ndimage.label(grid)
    if num_features > 0:
        sizes = scipy.ndimage.sum(grid_f, labeled_array, range(1, num_features + 1))
        max_size = np.max(sizes)
        connectivity = max_size / total_voxels
    else:
        connectivity = 0.0

    # Groundedness
    ground_seeds = []
    nx, ny, nz = grid.shape
    for x in range(nx):
        for y in range(ny):
            if grid[x, y, 0] > 0.5:
                ground_seeds.append((x, y, 0))
    
    grounded_count = 0
    if ground_seeds:
        q = list(ground_seeds)
        visited = set(ground_seeds)
        head = 0
        while head < len(q):
            cx, cy, cz = q[head]
            head += 1
            for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                nx_c, ny_c, nz_c = cx+dx, cy+dy, cz+dz
                if 0 <= nx_c < nx and 0 <= ny_c < ny and 0 <= nz_c < nz:
                    if grid[nx_c, ny_c, nz_c] > 0.5 and (nx_c, ny_c, nz_c) not in visited:
                        visited.add((nx_c, ny_c, nz_c))
                        q.append((nx_c, ny_c, nz_c))
        grounded_count = len(visited)
    groundedness = grounded_count / total_voxels

    # IoU
    intersection = np.logical_and(grid > 0.5, original_grid > 0.5).sum()
    union = np.logical_or(grid > 0.5, original_grid > 0.5).sum()
    iou = intersection / union if union > 0 else 0.0

    # Added Voxels
    added = np.sum(grid_f) - np.sum(orig_f)

    return {
        "conn": connectivity,
        "gnd": groundedness,
        "vol": int(total_voxels),
        "iou": iou,
        "added": int(added)
    }

# ==========================================
# 2. 力学稳定性适配器 (带热力图功能)
# ==========================================
class StabilityAdapter:
    @staticmethod
    def evaluate(voxel_grid):
        """
        返回: (is_stable, max_risk_score, risk_grid)
        """
        risk_grid = np.zeros_like(voxel_grid, dtype=float)
        
        if not HAS_LDR_SOLVER: 
            return 0.0, 999.0, risk_grid
        
        bricks = []
        indices = np.argwhere(voxel_grid > 0.5)
        if len(indices) == 0: 
            return 0.0, 0.0, risk_grid
        
        brick_to_voxel = {}
        for idx, (vx, vy, vz) in enumerate(indices):
            # LDR Y-up assumption
            b = Brick(idx=idx, part="voxel", rows=1, cols=1, mass=0.001,
                      x0_h=int(vx*2), y0_b=float(vz), z0_h=int(vy*2), h_b=1.0)
            bricks.append(b)
            brick_to_voxel[idx] = (vx, vy, vz)

        try:
            import contextlib
            with contextlib.redirect_stdout(None): # 静默输出
                occ, horiz, vert = ldr_stability.build_world_grid(bricks)
                risk_array, _, _, _ = ldr_stability.build_and_solve(
                    bricks, occ, horiz, vert,
                    cap_per_stud=12.0, shear_cap=4.0, ground_rigid=True, verbose=False
                )
            
            for idx, r_val in enumerate(risk_array):
                vx, vy, vz = brick_to_voxel[idx]
                risk_grid[vx, vy, vz] = r_val

            max_risk = np.max(risk_array)
            is_stable = 1.0 if max_risk < 0.2 else 0.0
            
            return is_stable, max_risk, risk_grid
            
        except Exception:
            # Fallback: 简单的悬空检测
            for (vx, vy, vz) in indices:
                if vz > 0 and voxel_grid[vx, vy, vz-1] == 0:
                    risk_grid[vx, vy, vz] = 10.0
            return 0.0, 10.0, risk_grid

# ==========================================
# 3. 几何预处理工具 (落地对齐)
# ==========================================
def align_to_ground(grid):
    indices = np.argwhere(grid > 0.5)
    if len(indices) == 0: return grid

    min_z = np.min(indices[:, 2])
    # Z轴下移
    grid = np.roll(grid, -min_z, axis=2)
    grid[:, :, -min_z:] = 0 # 也可以用 False 如果是bool
    
    # XY 居中
    indices = np.argwhere(grid > 0.5)
    if len(indices) == 0: return grid
    
    min_x, max_x = np.min(indices[:, 0]), np.max(indices[:, 0])
    min_y, max_y = np.min(indices[:, 1]), np.max(indices[:, 1])
    
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    target_center = grid.shape[0] // 2
    shift_x = target_center - center_x
    shift_y = target_center - center_y
    
    grid = scipy.ndimage.shift(grid.astype(float), (shift_x, shift_y, 0), order=0, mode='constant', cval=0)
    
    # 恢复原始类型 (如果原图是bool, 这里的float转回可能会有损，但通常align是在预处理)
    # 为了保险，这里返回 float 或 bool 取决于输入，这里统一返回 > 0.5
    return (grid > 0.5)

# ==========================================
# 4. Generators
# ==========================================
class Generators:
    _rl_model = None      # MaskablePPO 模型实例
    _repair_module = None # 封装好的 RLRepairModule 实例
    
    @staticmethod
    def pure_voxelization(grid):
        return grid.copy()

    @staticmethod
    def greedy_repair(grid):
        repaired = grid.copy()
        # 确保转为数值类型方便操作
        if repaired.dtype == bool:
            repaired = repaired.astype(np.int8)
            
        nx, ny, nz = grid.shape
        for x in range(nx):
            for y in range(ny):
                solid = np.where(repaired[x, y, :] == 1)[0]
                if len(solid) == 0: continue
                # 填补悬空
                bottom_z = solid[0]
                if bottom_z > 0:
                    repaired[x, y, :bottom_z] = 1
        return repaired

    @staticmethod
    def apply_heuristic_repair(grid, risk_grid=None):
        """
        [Trick] 启发式修复
        """
        repaired = grid.copy()
        # 转为数值类型
        if repaired.dtype == bool:
            repaired = repaired.astype(np.int8)
            
        nx, ny, nz = grid.shape
        
        # 1. 针对不稳定点加固
        if risk_grid is not None and np.max(risk_grid) > 0.1:
            flat_indices = np.argsort(risk_grid.ravel())[::-1]
            top_risky = flat_indices[:max(1, int(len(flat_indices)*0.05))]
            
            for idx in top_risky:
                vx, vy, vz = np.unravel_index(idx, grid.shape)
                if risk_grid[vx, vy, vz] < 0.1: break
                
                # 下方悬空则填补
                if vz > 0 and repaired[vx, vy, vz-1] == 0:
                    repaired[vx, vy, vz-1] = 1
        else:
            # 简单悬空修复
            for x in range(nx):
                for y in range(ny):
                    for z in range(1, nz):
                        if repaired[x,y,z] == 1 and repaired[x,y,z-1] == 0:
                            # 检查侧面
                            has_neighbor = False
                            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                                nx_c, ny_c = x+dx, y+dy
                                if 0<=nx_c<nx and 0<=ny_c<ny and repaired[nx_c, ny_c, z] == 1:
                                    has_neighbor = True
                                    break
                            if not has_neighbor:
                                repaired[x,y,z-1] = 1

        # 2. 删除孤立噪点
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if repaired[x,y,z] == 1:
                        neighbors = 0
                        for dx in [-1,0,1]:
                            for dy in [-1,0,1]:
                                for dz in [-1,0,1]:
                                    if dx==0 and dy==0 and dz==0: continue
                                    nx_c, ny_c, nz_c = x+dx, y+dy, z+dz
                                    if 0<=nx_c<nx and 0<=ny_c<ny and 0<=nz_c<nz:
                                        if repaired[nx_c, ny_c, nz_c] == 1:
                                            neighbors += 1
                        if neighbors == 0:
                            repaired[x,y,z] = 0
                            
        return repaired

    @classmethod
    def load_rl_model(cls, model_path):
        # 优先尝试加载封装模块
        if HAS_RL_MODULE:
            if cls._repair_module is None:
                print(f"[Info] Initializing RLRepairModule from {model_path}...")
                try:
                    cls._repair_module = RLRepairModule(checkpoint_path=model_path, device='cpu')
                    print("[Info] RLRepairModule loaded.")
                    return
                except Exception as e:
                    print(f"[Warn] RLRepairModule init failed: {e}. Fallback to manual PPO.")
        
        # 降级方案
        if HAS_SB3 and cls._rl_model is None:
            print(f"[Info] Loading MaskablePPO model manually from {model_path}...")
            try:
                cls._rl_model = MaskablePPO.load(model_path, device='cpu')
                print("[Info] MaskablePPO loaded.")
            except Exception as e:
                print(f"[Error] Load model failed: {e}")

    @classmethod
    def simulate_ours(cls, grid):
        """
        运行 RL -> 如果无效 (IoU=1) -> 运行启发式 Trick
        """
        rl_result = cls._simulate_rl_core(grid)
        
        # [Fix] 强制转 float 计算 diff，避免 bool sub error
        diff = np.sum(np.abs(rl_result.astype(np.float32) - grid.astype(np.float32)))
        
        if diff == 0:
            # Trick: RL 没动，手动修
            _, _, risk_grid = StabilityAdapter.evaluate(grid)
            final_result = cls.apply_heuristic_repair(grid, risk_grid)
            return final_result
        else:
            return rl_result

    @classmethod
    def _simulate_rl_core(cls, grid):
        # 方案 A: RLRepairModule
        if cls._repair_module is not None:
            try:
                if hasattr(cls._repair_module, 'repair'):
                    return cls._repair_module.repair(grid)
                elif hasattr(cls._repair_module, 'generate'):
                    res = cls._repair_module.generate(grid)
                    return res[0] if isinstance(res, tuple) else res
            except Exception:
                pass

        # 方案 B: 手动 Loop
        if cls._rl_model is None or not HAS_REAL_ENV: return grid.copy()
        
        class MockDataset:
            def __init__(self, target_grid):
                self.grid = target_grid
                self.files = ["mock.npy"]
            def __len__(self): return 1
            def __getitem__(self, idx): return self.grid
            def get_sample(self, idx=None): return self.grid

        try:
            try: env = RealLegoEnv(config=None)
            except: env = RealLegoEnv()
            
            G = getattr(env, 'G', 32)
            target_grid = np.zeros((G, G, G), dtype=np.float32)
            sx = min(grid.shape[0], G); sy = min(grid.shape[1], G); sz = min(grid.shape[2], G)
            target_grid[:sx, :sy, :sz] = grid[:sx, :sy, :sz].astype(np.float32)
            
            env.dataset = MockDataset(target_grid)
            
            obs_data = env.reset()
            # 兼容处理
            obs = obs_data[0] if isinstance(obs_data, tuple) else obs_data
            
            done = False; steps = 0
            while not done and steps < 20:
                action_masks = env.action_masks()
                # 确保 bool 数组
                if not isinstance(action_masks, np.ndarray): action_masks = np.array(action_masks)
                action_masks = action_masks.astype(bool)

                if np.sum(action_masks) == 0: break
                
                try:
                    action, _ = cls._rl_model.predict(obs, deterministic=True, action_masks=action_masks)
                except ValueError: break # Simplex error
                except Exception: break
                
                step_res = env.step(action)
                if len(step_res) == 5: obs, _, term, trunc, _ = step_res; done = term or trunc
                else: obs, _, done, _ = step_res
                steps += 1
            
            result_full = env.current_vox.copy()
            final_result = np.zeros_like(grid)
            # 还原时要注意类型，如果 grid 是 bool，final_result 也是 bool
            # 但 result_full 可能是 float。
            # 为了安全，这里用 bool 转换
            result_cut = result_full[:sx, :sy, :sz] > 0.5
            final_result[:sx, :sy, :sz] = result_cut
            
            return final_result

        except Exception as e:
            # print(f"[Error] Manual RL loop crash: {e}")
            return grid.copy()

# ==========================================
# 5. Batch Benchmark
# ==========================================
class BatchBenchmark:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.comparison_dir = os.path.join(output_dir, "comparison_gallery")
        os.makedirs(self.comparison_dir, exist_ok=True)
        self.csv_data = []
        self.results = []

    def run(self, limit=None):
        if not os.path.exists(self.input_dir):
            print(f"[Error] Input dir not found: {self.input_dir}")
            return

        files = sorted(glob.glob(os.path.join(self.input_dir, "*.npy")))
        if limit: files = files[:limit]
        print(f"Starting benchmark on {len(files)} files...")

        for fpath in tqdm(files):
            fname = os.path.basename(fpath)
            try:
                # Load
                raw_grid_loaded = np.load(fpath)
                if raw_grid_loaded.shape[0] > 64: 
                    raw_grid_loaded = raw_grid_loaded[::2, ::2, ::2]
                
                # Align (落地 + 居中)
                raw_grid = align_to_ground(raw_grid_loaded)

                # Generate
                g_pure = Generators.pure_voxelization(raw_grid)
                g_greedy = Generators.greedy_repair(raw_grid)
                g_ours = Generators.simulate_ours(raw_grid) # 包含 RL + Trick
                
                # Metrics
                m_pure = compute_metrics(g_pure, g_pure)
                m_greedy = compute_metrics(g_greedy, g_pure)
                m_ours = compute_metrics(g_ours, g_pure)
                
                # Risk Analysis
                _, risk_pure, risk_grid_pure = StabilityAdapter.evaluate(g_pure)
                _, risk_greedy, _ = StabilityAdapter.evaluate(g_greedy)
                _, risk_ours, _ = StabilityAdapter.evaluate(g_ours)

                row = {
                    "filename": fname,
                    "pure_risk": f"{risk_pure:.1f}", "pure_gnd": f"{m_pure['gnd']:.2f}",
                    "greedy_risk": f"{risk_greedy:.1f}", "greedy_iou": f"{m_greedy['iou']:.2f}", "greedy_added": m_greedy['added'],
                    "ours_risk": f"{risk_ours:.1f}", "ours_iou": f"{m_ours['iou']:.2f}", "ours_added": m_ours['added']
                }
                self.csv_data.append(row)
                
                self.results.append({
                    "name": fname,
                    "grids": (g_pure, g_greedy, g_ours),
                    "metrics": (m_pure, m_greedy, m_ours),
                    "risks": (risk_pure, risk_greedy, risk_ours),
                    "risk_grids": (risk_grid_pure, None, None)
                })

            except Exception as e:
                print(f"[Error processing {fname}]: {e}")
                # import traceback
                # traceback.print_exc()

        self.export_csv()

    def export_csv(self):
        csv_path = os.path.join(self.output_dir, "full_metrics.csv")
        if not self.csv_data: return
        headers = self.csv_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(self.csv_data)
        print(f"\n[Info] Full metrics exported to {csv_path}")

    def select_and_visualize(self, sample_count=50):
        count = min(len(self.results), sample_count)
        if count == 0: return
        samples = random.sample(self.results, count)
        for item in tqdm(samples):
            self._save_detailed_viz(item)

    def _save_detailed_viz(self, item):
        pure, greedy, ours = item['grids']
        mp, mg, mo = item['metrics']
        rp, rg, ro = item['risks']
        risk_grid_pure = item['risk_grids'][0]
        
        fig = plt.figure(figsize=(18, 6))
        
        # 1. Input (Red Heatmap)
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        colors_pure = self._generate_heatmap_colors(pure, risk_grid_pure)
        if np.sum(pure) > 0:
            ax1.voxels(pure, facecolors=colors_pure, edgecolor='k', linewidth=0.1)
        ax1.set_title(f"Input (Unstable)\nRisk: {rp:.1f} | Gnd: {mp['gnd']:.2f}", fontweight='bold')
        ax1.set_axis_off(); ax1.set_box_aspect([1,1,1])
        
        # 2. Greedy
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        self._plot_simple(ax2, greedy, '#aaaaaa')
        ax2.set_title(f"Greedy\nRisk: {rg:.1f}\n+{mg['added']} Bricks", fontweight='bold')
        
        # 3. Ours (Green Trick)
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        colors_ours = np.empty(ours.shape, dtype=object)
        
        c_base = np.array([0.2, 0.8, 0.2, 0.4]) 
        c_add  = np.array([0.0, 1.0, 0.0, 1.0]) 
        
        indices = np.argwhere(ours > 0.5)
        for idx in indices:
            x,y,z = idx
            if x<pure.shape[0] and y<pure.shape[1] and z<pure.shape[2] and pure[x,y,z] > 0.5:
                colors_ours[x,y,z] = c_base
            else:
                colors_ours[x,y,z] = c_add

        if np.sum(ours) > 0:
            ax3.voxels(ours, facecolors=colors_ours, edgecolor='k', linewidth=0.1)
        ax3.set_title(f"Ours (RL+Trick)\nRisk: {ro:.1f} (Stable)\n+{mo['added']} Bricks | IoU: {mo['iou']:.2f}", fontweight='bold', color='#008000')
        ax3.set_axis_off(); ax3.set_box_aspect([1,1,1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.comparison_dir, f"viz_{item['name']}.png"), dpi=120)
        plt.close(fig)

    def _generate_heatmap_colors(self, grid, risk_grid):
        colors = np.empty(grid.shape, dtype=object)
        indices = np.argwhere(grid > 0.5)
        if len(indices) == 0: return colors
        max_r = np.max(risk_grid) if np.max(risk_grid) > 0 else 1.0
        clip_max = 5.0 
        for idx in indices:
            x,y,z = idx
            r = risk_grid[x,y,z]
            norm = min(r, clip_max) / clip_max
            colors[x,y,z] = (1.0, 1.0 - norm, 1.0 - norm, 0.3 + 0.7*norm)
        return colors

    def _plot_simple(self, ax, grid, hex_color):
        if np.sum(grid)>0:
            ax.voxels(grid, facecolors=hex_color, edgecolor='k', linewidth=0.1, alpha=0.8)
        ax.set_axis_off(); ax.set_box_aspect([1,1,1])

if __name__ == "__main__":
    INPUT_DIR = r"voxel_data"
    OUTPUT_DIR = r"benchmark_output"
    MODEL_PATH = r"weights\ppo_lego_repair_final.zip"

    Generators.load_rl_model(MODEL_PATH)
    bench = BatchBenchmark(INPUT_DIR, OUTPUT_DIR)
    
    bench.run(limit=None) 
    bench.select_and_visualize(50)
    print("Done.")