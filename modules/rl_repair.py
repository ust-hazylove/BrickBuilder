import os

import numpy as np
from sb3_contrib import MaskablePPO

from modules.train_utils import Lego3DCNN  # noqa: F401 - needed for model deserialization
from ppo_repair.Env import EnvConfig, LegoVoxelRepairEnv


class RLRepairModule:
    def __init__(self, checkpoint_path, device="cuda"):
        print(f"--- Loading RL Repair Agent from {checkpoint_path} ---")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RL Model not found: {checkpoint_path}")

        self.device = device
        self.model = MaskablePPO.load(
            checkpoint_path,
            device=device,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.1,
            },
        )
        self.risk_threshold = 0.9

    def inference(self, initial_voxels: np.ndarray, risk_hints=None, max_steps=50):
        grid_size = 16
        voxels = self._fit_to_grid(initial_voxels, grid_size)
        env = LegoVoxelRepairEnv(
            config=EnvConfig(
                grid_size=grid_size,
                max_steps=min(max_steps, 24),
                risk_threshold=self.risk_threshold,
                use_clip_reward=False,
            )
        )

        obs, _ = env.reset()
        env.current_vox = voxels.astype(np.uint8)
        if risk_hints:
            self._apply_risk_hints(env.current_vox, risk_hints)

        unstable_mask = env._get_unstable_mask()
        env.last_unstable_count = int(unstable_mask.sum())
        obs = env._make_observation(env.current_vox, unstable_mask)

        done = False
        step = 0
        while not done and step < max_steps:
            action_masks = env.action_masks()
            action, _states = self.model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            if truncated or info.get("unstable_count", 0) == 0:
                break

        return env.current_vox > 0

    def _fit_to_grid(self, initial_voxels: np.ndarray, grid_size: int):
        if initial_voxels.shape == (grid_size, grid_size, grid_size):
            return initial_voxels

        temp = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        src_x = min(grid_size, initial_voxels.shape[0])
        src_y = min(grid_size, initial_voxels.shape[1])
        src_z = min(grid_size, initial_voxels.shape[2])
        x0 = (grid_size - src_x) // 2
        y0 = (grid_size - src_y) // 2
        z0 = (grid_size - src_z) // 2
        temp[x0:x0 + src_x, y0:y0 + src_y, z0:z0 + src_z] = initial_voxels[:src_x, :src_y, :src_z]
        return temp

    def _apply_risk_hints(self, voxels: np.ndarray, risk_hints):
        for hint in risk_hints[:24]:
            grid_pos = hint.get("grid_pos")
            size = hint.get("size")
            if grid_pos is None or size is None:
                continue
            x, y, z = [int(v) for v in grid_pos]
            dx, dy = [int(v) for v in size]
            cx = min(max(x + dx // 2, 0), voxels.shape[0] - 1)
            cy = min(max(y + dy // 2, 0), voxels.shape[1] - 1)

            for zz in range(min(z, voxels.shape[2] - 1), -1, -1):
                voxels[cx, cy, zz] = 1

            for px in range(max(0, x), min(voxels.shape[0], x + dx)):
                for py in range(max(0, y), min(voxels.shape[1], y + dy)):
                    voxels[px, py, min(z, voxels.shape[2] - 1)] = 1
