import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.brick_mapper import BrickMapper
from modules.high_risk_predictor import HighRiskPredictor
from modules.risk_analysis import compute_rule_risk_masks, detect_risky_bricks
from modules.semantic_clip import SemanticCLIPScorer
from ppo_repair.ldr_dataset import iter_dataset_files, load_ldr_as_voxels


@dataclass
class EnvConfig:
    grid_size: int = 16
    max_steps: int = 24
    dataset_root: str = "npy_raw"
    stability_reward: float = 0.2
    iou_penalty: float = 0.01
    step_penalty: float = -0.01
    risk_checkpoint: str = ""
    risk_threshold: float = 0.8
    use_risk_predictor: bool = True
    risk_reward: float = 1.5
    clip_reward: float = 0.75
    terminal_bonus: float = 2.0
    use_clip_reward: bool = True


class VoxelDataset:
    def __init__(self, root_dir: str, grid_size: int = 32):
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.files = list(iter_dataset_files(root_dir))
        if self.files:
            print(f"[Dataset] Found {len(self.files)} models in {root_dir}.")
        else:
            print(f"[Dataset] No training samples found in {root_dir}.")

    def get_sample(self) -> Optional[np.ndarray]:
        if not self.files:
            return None

        path = self.files[np.random.randint(0, len(self.files))]
        try:
            if path.endswith(".ldr"):
                data = load_ldr_as_voxels(path, grid_size=self.grid_size)
            else:
                payload = np.load(path)
                if path.endswith(".npz"):
                    key = "voxels" if "voxels" in payload else list(payload.keys())[0]
                    data = payload[key]
                else:
                    data = payload
            return self._fit_to_grid((data > 0).astype(np.uint8)), self._build_meta(path)
        except Exception as exc:
            print(f"[Dataset] Failed to load {path}: {exc}")
            return None

    def _build_meta(self, path: str) -> Dict[str, str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        prompt = stem
        while prompt and prompt[-1].isdigit():
            prompt = prompt[:-1]
        prompt = prompt.replace("_s", " ").replace("_rotate_x", "")
        prompt = prompt.replace("_", " ").strip()
        return {"path": path, "prompt": prompt or "lego object"}

    def _fit_to_grid(self, grid: np.ndarray) -> np.ndarray:
        if grid.shape == (self.grid_size, self.grid_size, self.grid_size):
            return grid

        fitted = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        src = np.argwhere(grid > 0)
        if src.size == 0:
            return fitted

        mins = src.min(axis=0)
        maxs = src.max(axis=0) + 1
        cropped = grid[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        dims = np.array(cropped.shape, dtype=int)
        if np.any(dims > self.grid_size):
            from ppo_repair.ldr_dataset import fit_voxels_to_grid

            return fit_voxels_to_grid(cropped, grid_size=self.grid_size)

        starts = ((self.grid_size - dims) // 2).astype(int)
        fitted[
            starts[0]:starts[0] + dims[0],
            starts[1]:starts[1] + dims[1],
            starts[2]:starts[2] + dims[2],
        ] = cropped
        return fitted


class LegoVoxelRepairEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.dataset = VoxelDataset(self.cfg.dataset_root, self.cfg.grid_size)
        self.mapper = BrickMapper()
        self.risk_predictor = self._load_risk_predictor()

        g = self.cfg.grid_size
        self.G = g
        self.G2 = g * g
        self.G3 = g * g * g

        self.num_channels = 2
        self.observation_space = spaces.Box(0.0, 1.0, (self.num_channels, g, g, g), dtype=np.float32)
        self.action_space = spaces.Discrete(3 * self.G3)

        self.current_vox: Optional[np.ndarray] = None
        self.original_vox: Optional[np.ndarray] = None
        self.current_meta: Dict[str, Any] = {}
        self.step_count = 0
        self.last_unstable_count = 0
        self.last_risk_count = 0
        self.last_clip_score = 0.0
        self._unstable_cache_vox: Optional[np.ndarray] = None
        self._unstable_cache_mask: Optional[np.ndarray] = None
        self._risk_cache: Optional[Dict[str, Any]] = None
        self.clip_scorer = self._load_clip_scorer()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.step_count = 0

        sample = self.dataset.get_sample()
        if sample is None:
            sample = np.zeros((self.G, self.G, self.G), dtype=np.uint8)
            sample[max(1, self.G // 3):max(2, self.G // 2), max(1, self.G // 3):max(2, self.G // 2), max(1, self.G // 3):max(2, self.G // 2)] = 1
            meta = {"path": "", "prompt": "lego object"}
        else:
            sample, meta = sample

        self.current_vox = sample.copy()
        self.original_vox = sample.copy()
        self.current_meta = meta
        self._invalidate_cache()

        unstable_mask = self._get_unstable_mask()
        self.last_unstable_count = int(unstable_mask.sum())
        self.last_risk_count = self._get_risk_count()
        self.last_clip_score = self._get_clip_score()
        return self._make_observation(self.current_vox, unstable_mask), {"unstable": self.last_unstable_count}

    def step(self, action: int):
        action_type, x, y, z = self.decode_action(int(action))
        self.step_count += 1

        added_voxels = 0
        if action_type == 1:
            added_voxels = self._apply_bridge_trust(x, y, z)
        elif action_type == 2:
            added_voxels = self._apply_support_trust(x, y, z)

        new_unstable_mask = self._get_unstable_mask()
        new_unstable_count = int(new_unstable_mask.sum())
        delta_stable = self.last_unstable_count - new_unstable_count
        new_risk_count = self._get_risk_count()
        delta_risk = self.last_risk_count - new_risk_count
        new_clip_score = self._get_clip_score()
        delta_clip = new_clip_score - self.last_clip_score

        reward = (
            delta_stable * self.cfg.stability_reward
            + delta_risk * self.cfg.risk_reward
            + delta_clip * self.cfg.clip_reward
            + added_voxels * -self.cfg.iou_penalty
            + self.cfg.step_penalty
        )

        self.last_unstable_count = new_unstable_count
        self.last_risk_count = new_risk_count
        self.last_clip_score = new_clip_score

        terminated = False
        if new_unstable_count == 0:
            terminated = True
            reward += self.cfg.terminal_bonus

        truncated = self.step_count >= self.cfg.max_steps
        info = {
            "unstable_count": new_unstable_count,
            "delta_stable": delta_stable,
            "risk_count": new_risk_count,
            "delta_risk": delta_risk,
            "clip_score": round(float(new_clip_score), 4),
            "delta_clip": round(float(delta_clip), 4),
            "added_voxels": added_voxels,
            "valid_action": 1.0,
        }
        return self._make_observation(self.current_vox, new_unstable_mask), float(reward), terminated, truncated, info

    def _load_risk_predictor(self) -> Optional[HighRiskPredictor]:
        if not self.cfg.use_risk_predictor:
            return None
        if not self.cfg.risk_checkpoint:
            return None
        if not os.path.exists(self.cfg.risk_checkpoint):
            print(f"[Risk] Checkpoint not found: {self.cfg.risk_checkpoint}")
            return None
        try:
            return HighRiskPredictor(self.cfg.risk_checkpoint)
        except Exception as exc:
            print(f"[Risk] Failed to initialize predictor: {exc}")
            return None

    def _load_clip_scorer(self):
        if not self.cfg.use_clip_reward:
            return None
        try:
            return SemanticCLIPScorer(device="cuda")
        except Exception as exc:
            print(f"[CLIP] Failed to initialize scorer: {exc}")
            return None

    def _make_observation(self, vox: np.ndarray, unstable_mask: np.ndarray) -> np.ndarray:
        obs = np.zeros((self.num_channels, self.G, self.G, self.G), dtype=np.float32)
        obs[0] = vox.astype(np.float32)
        obs[1] = unstable_mask.astype(np.float32)
        return obs

    def _get_unstable_mask(self) -> np.ndarray:
        vox = self.current_vox
        if vox.sum() == 0:
            return np.zeros_like(vox, dtype=bool)
        if self._unstable_cache_vox is not None and np.array_equal(vox, self._unstable_cache_vox):
            return self._unstable_cache_mask.copy()
        risk_payload = self._compute_risk_payload(vox)
        unstable_mask = risk_payload["mask"]

        self._unstable_cache_vox = vox.copy()
        self._unstable_cache_mask = unstable_mask.copy()
        return unstable_mask

    def _compute_risk_payload(self, voxel_grid: np.ndarray) -> Dict[str, Any]:
        if self._risk_cache is not None and np.array_equal(voxel_grid, self._risk_cache["voxels"]):
            return self._risk_cache
        try:
            brick_list = self.mapper.map_voxels_to_bricks(voxel_grid, verbose=False)
            risky, rule_stats, risk_source = detect_risky_bricks(
                voxel_grid,
                brick_list,
                risk_predictor=self.risk_predictor,
                risk_threshold=self.cfg.risk_threshold,
            )
        except Exception as exc:
            print(f"[Risk] Prediction failed: {exc}")
            payload = {
                "voxels": voxel_grid.copy(),
                "mask": np.zeros_like(voxel_grid, dtype=bool),
                "risky": [],
                "rule_stats": {"rule_risk_voxels": 0, "floating_voxels": 0, "unsupported_voxels": 0, "isolated_voxels": 0, "rule_components": 0},
                "source": "error",
            }
            self._risk_cache = payload
            return payload

        risk_mask = np.zeros_like(voxel_grid, dtype=bool)
        rule_masks = compute_rule_risk_masks(voxel_grid)
        risk_mask |= rule_masks["rule_mask"]
        for item in risky:
            grid_pos = item.get("grid_pos")
            size = item.get("size")
            if grid_pos is None or size is None:
                continue
            x, y, z = grid_pos
            dx, dy = size
            x1 = min(self.G, x + dx)
            y1 = min(self.G, y + dy)
            z1 = min(self.G, z + 1)
            risk_mask[x:x1, y:y1, z:z1] = True
            if z > 0:
                risk_mask[x:x1, y:y1, :z] |= (voxel_grid[x:x1, y:y1, :z] > 0)
        payload = {
            "voxels": voxel_grid.copy(),
            "mask": risk_mask,
            "risky": risky,
            "rule_stats": rule_stats,
            "source": risk_source,
        }
        self._risk_cache = payload
        return payload

    def _get_risk_count(self) -> int:
        return int(len(self._compute_risk_payload(self.current_vox)["risky"]))

    def _get_clip_score(self) -> float:
        if self.clip_scorer is None:
            return 0.0
        prompt = self.current_meta.get("prompt")
        try:
            return float(self.clip_scorer.score_voxel_similarity(self.current_vox, self.original_vox, text_prompt=prompt))
        except Exception as exc:
            print(f"[CLIP] Scoring failed: {exc}")
            return 0.0

    def action_masks(self) -> np.ndarray:
        mask = np.zeros((3, self.G, self.G, self.G), dtype=bool)
        vox = self.current_vox.astype(bool)
        empty = ~vox

        unstable_vox = self._get_unstable_mask()
        if not np.any(unstable_vox):
            mask[0, 0, 0, 0] = True
            return mask.flatten()

        struct = generate_binary_structure(rank=3, connectivity=1)
        neighbors = binary_dilation(unstable_vox, structure=struct) & empty
        mask[1] = neighbors

        below_unstable = np.zeros_like(vox)
        below_unstable[:, :, :-1] = unstable_vox[:, :, 1:]
        mask[2] = below_unstable & empty

        mask[0] = unstable_vox
        if not np.any(mask):
            mask[0, 0, 0, 0] = True
        return mask.flatten()

    def _apply_bridge_trust(self, x: int, y: int, z: int) -> int:
        self._invalidate_cache()
        count = 0
        if self.current_vox[x, y, z] == 0:
            self.current_vox[x, y, z] = 1
            count += 1
        if x > 0 and self.current_vox[x - 1, y, z] == 0:
            self.current_vox[x - 1, y, z] = 1
            count += 1
        if x < self.G - 1 and self.current_vox[x + 1, y, z] == 0:
            self.current_vox[x + 1, y, z] = 1
            count += 1
        return count

    def _apply_support_trust(self, x: int, y: int, z: int) -> int:
        self._invalidate_cache()
        count = 0
        current_z = z
        while current_z >= 0:
            if self.current_vox[x, y, current_z] > 0:
                break
            self.current_vox[x, y, current_z] = 1
            count += 1
            current_z -= 1
        return count

    def decode_action(self, idx: int):
        act_type = idx // self.G3
        rem = idx % self.G3
        x = rem // self.G2
        rem = rem % self.G2
        y = rem // self.G
        z = rem % self.G
        return act_type, x, y, z

    def _invalidate_cache(self):
        self._unstable_cache_vox = None
        self._unstable_cache_mask = None
        self._risk_cache = None
