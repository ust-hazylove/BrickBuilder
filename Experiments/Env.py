import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import glob
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
from scipy.ndimage import label, generate_binary_structure, binary_dilation

# ==============================================================================
# 1. 配置与数据集
# ==============================================================================
@dataclass
class EnvConfig:
    grid_size: int = 32          # 体素分辨率
    max_steps: int = 32          # 最大修复步数
    dataset_root: str = "npy_raw"
    
    # 奖励权重
    # 逻辑: 核心目标是消除 unstable (conn/ground)，同时尽量少改动原结构 (iou/step)
    stability_reward: float = 0.2  # 每消除一个不稳定体素的奖励
    iou_penalty: float = 0.01       # 每增加一个体素(改变原状)的惩罚
    step_penalty: float = -0.01    # 步数惩罚

class VoxelDataset:
    """简单的体素数据加载器"""
    def __init__(self, root_dir: str, grid_size: int = 32):
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.files = []
        if root_dir and os.path.exists(root_dir):
            self.files.extend(glob.glob(os.path.join(root_dir, "*.npy")))
            self.files.extend(glob.glob(os.path.join(root_dir, "*.npz")))
            print(f"[Dataset] Found {len(self.files)} models.")
        
    def get_sample(self) -> np.ndarray:
        if not self.files: return None
        idx = np.random.randint(0, len(self.files))
        try:
            data = np.load(self.files[idx])
            if self.files[idx].endswith('.npz'):
                data = data[list(data.keys())[0]]
            # 尺寸适配
            if data.shape != (self.grid_size, self.grid_size, self.grid_size):
                # 简单裁剪或填充，这里略过复杂resize
                pass
            return (data > 0).astype(np.uint8)
        except:
            return None

# ==============================================================================
# 2. 核心环境
# ==============================================================================
class LegoVoxelRepairEnv(gym.Env):
    """
    自监督结构修复环境
    Task: 给定一个不稳定的体素结构，通过添加最少的支撑/连接，使其变得稳定。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.dataset = VoxelDataset(self.cfg.dataset_root, self.cfg.grid_size)
        
        G = self.cfg.grid_size
        self.G = G
        self.G2 = G * G
        self.G3 = G * G * G

        # ---- Observation: [2, G, G, G] ----
        # Channel 0: Current Voxel (当前结构)
        # Channel 1: Unstable Mask (诊断出的病灶区域)
        self.num_channels = 2
        self.observation_space = spaces.Box(0.0, 1.0, (self.num_channels, G, G, G), dtype=np.float32)

        # ---- Action: Discrete(3 * G^3) ----
        # 0:Merge, 1:Bridge, 2:Support (Flattened)
        self.action_space = spaces.Discrete(3 * self.G3)

        # Runtime State
        self.current_vox: np.ndarray = None  # 当前正在修的
        self.original_vox: np.ndarray = None # 原始设计图 (用于计算 IoU 偏差)
        self.step_count = 0
        
        # Metrics Cache
        self.last_unstable_count = 0

    # --------------------------------------------------------------------------
    # Core API
    # --------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # 1. 加载数据
        sample = self.dataset.get_sample()
        if sample is None:
            # Fallback Mock Data (悬空结构)
            sample = np.zeros((self.G, self.G, self.G), dtype=np.uint8)
            sample[10:15, 10:15, 10:15] = 1 # 悬空立方体
        
        self.current_vox = sample.copy()
        self.original_vox = sample.copy() # 记录原始形状作为 Reference
        
        # 2. 初始诊断
        unstable_mask = self._get_unstable_mask()
        self.last_unstable_count = unstable_mask.sum()
        
        return self._make_observation(self.current_vox, unstable_mask), {"unstable": self.last_unstable_count}

    def step(self, action: int):
        # 1. Decode Action
        action_type, x, y, z = self.decode_action(int(action))
        self.step_count += 1
        
        # 2. Execute (Trust-based)
        # MaskablePPO 保证了动作都在 Unstable 区域附近，直接执行
        added_voxels = 0
        if action_type == 0: # Merge
            # Merge 逻辑上是改变图连接，体素不变，但在修复任务中可以视为"加固"
            pass 
        elif action_type == 1: # Bridge
            added_voxels = self._apply_bridge_trust(x, y, z)
        elif action_type == 2: # Support
            added_voxels = self._apply_support_trust(x, y, z)
            
        # 3. New Diagnosis
        new_unstable_mask = self._get_unstable_mask()
        new_unstable_count = new_unstable_mask.sum()
        
        # 4. Calculate Reward
        # 奖励 = (修复了多少个不稳定块) * 权重 - (用了多少砖) * 惩罚
        delta_stable = self.last_unstable_count - new_unstable_count
        
        reward = (delta_stable * self.cfg.stability_reward) + \
                 (added_voxels * -self.cfg.iou_penalty) + \
                 self.cfg.step_penalty

        # Update Cache
        self.last_unstable_count = new_unstable_count
        
        # 5. Termination
        terminated = False
        if new_unstable_count == 0:
            terminated = True
            reward += 1.0 # 通关奖励
            
        truncated = (self.step_count >= self.cfg.max_steps)
        
        info = {
            "unstable_count": new_unstable_count,
            "delta_stable": delta_stable,
            "added_voxels": added_voxels,
            "valid_action": 1.0 # Always valid with masking
        }
        
        return self._make_observation(self.current_vox, new_unstable_mask), float(reward), terminated, truncated, info

    # --------------------------------------------------------------------------
    # Observation & Diagnosis
    # --------------------------------------------------------------------------
    def _make_observation(self, vox: np.ndarray, unstable_mask: np.ndarray) -> np.ndarray:
        obs = np.zeros((self.num_channels, self.G, self.G, self.G), dtype=np.float32)
        obs[0] = vox.astype(np.float32)
        obs[1] = unstable_mask.astype(np.float32) # 告诉 Agent 哪里坏了
        return obs

    def _get_unstable_mask(self) -> np.ndarray:
        """
        物理诊断核心: 找出所有没有接地的体素 (Floating/Unstable)
        """
        vox = self.current_vox
        if vox.sum() == 0: return np.zeros_like(vox, dtype=bool)

        # 1. 连通分量分析 (6-neighbor)
        struct = generate_binary_structure(rank=3, connectivity=1)
        labeled, num = label(vox > 0, structure=struct)
        
        # 2. 找出接地 (y=0) 的 ID
        ground_ids = np.unique(labeled[:, 0, :])
        ground_ids = ground_ids[ground_ids != 0]
        
        # 3. 生成 Mask: 是砖块 BUT 不是接地的
        is_grounded = np.isin(labeled, ground_ids)
        unstable_mask = (vox > 0) & (~is_grounded)
        
        return unstable_mask

    # --------------------------------------------------------------------------
    # Action Masking (Targeted Repair)
    # --------------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """
        Action Masking: 只允许在不稳定区域附近操作
        """
        mask = np.zeros((3, self.G, self.G, self.G), dtype=bool)
        vox = self.current_vox.astype(bool)
        empty = ~vox
        
        # 1. 获取病灶
        unstable_vox = self._get_unstable_mask()
        
        # 如果已经修好了(全稳定)，允许 No-Op 防止报错
        if not np.any(unstable_vox):
            mask[0, 0, 0, 0] = True
            return mask.flatten()
            
        # 2. 找到不稳定体素的邻域 (Dilate)
        struct = generate_binary_structure(rank=3, connectivity=1)
        # 膨胀一步，找到紧挨着不稳定块的空位
        neighbors = binary_dilation(unstable_vox, structure=struct) & empty
        
        # --- Mask 1: Bridge ---
        # 策略: 允许在不稳定块的邻域搭桥
        mask[1] = neighbors
        
        # --- Mask 2: Support ---
        # 策略: 允许在不稳定块的正下方(y-1)建造支撑
        # unstable_vox 的 y 移到 y-1 的位置
        below_unstable = np.zeros_like(vox)
        below_unstable[:, :-1, :] = unstable_vox[:, 1:, :] 
        
        mask[2] = below_unstable & empty
        
        # --- Mask 0: Merge ---
        # 策略: 仅允许在不稳定块本身做 Merge
        mask[0] = unstable_vox

        # Safety Valve
        if not np.any(mask):
            mask[0, 0, 0, 0] = True
            
        return mask.flatten()

    # --------------------------------------------------------------------------
    # Trust-Based Actions
    # --------------------------------------------------------------------------
    def _apply_bridge_trust(self, x, y, z) -> int:
        """简单的 Bridge: 填补当前位 + 尝试向两边扩展"""
        count = 0
        if self.current_vox[x, y, z] == 0:
            self.current_vox[x, y, z] = 1
            count += 1
        
        # 简单的启发式: 尝试往 X 轴连
        if x > 0 and self.current_vox[x-1, y, z] == 0:
            self.current_vox[x-1, y, z] = 1; count += 1
        if x < self.G-1 and self.current_vox[x+1, y, z] == 0:
            self.current_vox[x+1, y, z] = 1; count += 1
            
        return count

    def _apply_support_trust(self, x, y, z) -> int:
        """简单的 Support: 向下射线填充"""
        count = 0
        curr_y = y
        while curr_y >= 0:
            if self.current_vox[x, curr_y, z] > 0:
                break
            self.current_vox[x, curr_y, z] = 1
            count += 1
            curr_y -= 1
        return count

    def decode_action(self, idx):
        act_type = idx // self.G3
        rem = idx % self.G3
        x = rem // self.G2
        rem = rem % self.G2
        y = rem // self.G
        z = rem % self.G
        return act_type, x, y, z