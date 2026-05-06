import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import gymnasium as gym
except ImportError:
    import gym

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Env import EnvConfig, LegoVoxelRepairEnv
from ldr_dataset import prepare_ldr_dataset


class LegoMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "unstable_count" in info:
                self.logger.record("custom/unstable_count", info["unstable_count"])
            if "risk_count" in info:
                self.logger.record("custom/risk_count", info["risk_count"])
            if "clip_score" in info:
                self.logger.record("custom/clip_score", info["clip_score"])
            if "added_voxels" in info:
                self.logger.record("custom/added_voxels", info["added_voxels"])
            if "delta_stable" in info:
                self.logger.record("custom/delta_stable", info["delta_stable"])
            if "delta_risk" in info:
                self.logger.record("custom/delta_risk", info["delta_risk"])
            if "delta_clip" in info:
                self.logger.record("custom/delta_clip", info["delta_clip"])
            if "valid_action" in info:
                self.logger.record("custom/valid_action_rate", info["valid_action"])
        return True


class Lego3DCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        for module in self.cnn.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.LayerNorm(n_flatten),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_wrapper_attr("action_masks")()


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for Lego repair with high-risk guidance.")
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_root", type=str, default="ppo_repair/data/train_voxels_16")
    parser.add_argument("--source_ldr_root", type=str, default="data/source_ldr")
    parser.add_argument("--risk_checkpoint", type=str, default="weights/high_risk_predictor_styled_best.pt")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def ensure_dataset(args) -> str:
    dataset_root = os.path.abspath(args.dataset_root)
    os.makedirs(dataset_root, exist_ok=True)
    has_voxel_files = any(name.endswith((".npy", ".npz")) for name in os.listdir(dataset_root))
    if has_voxel_files:
        print(f"[Dataset] Reusing prepared voxel dataset: {dataset_root}")
        return dataset_root

    if not os.path.isdir(args.source_ldr_root):
        raise FileNotFoundError(f"Cannot find source LDR dataset: {args.source_ldr_root}")

    print(f"[Dataset] Preparing 16^3 voxel dataset from {args.source_ldr_root}")
    converted = prepare_ldr_dataset(args.source_ldr_root, dataset_root, grid_size=16, limit=args.limit)
    if converted <= 0:
        raise RuntimeError("No LDR files were converted into PPO training samples.")
    print(f"[Dataset] Prepared {converted} samples at {dataset_root}")
    return dataset_root


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    os.makedirs(args.log_dir, exist_ok=True)

    dataset_root = ensure_dataset(args)
    if args.prepare_only:
        return

    config = EnvConfig(
        dataset_root=dataset_root,
        grid_size=16,
        max_steps=24,
        stability_reward=2.0,
        iou_penalty=0.1,
        step_penalty=-0.01,
        risk_checkpoint=args.risk_checkpoint,
        risk_threshold=0.8,
        use_risk_predictor=True,
        risk_reward=1.5,
        clip_reward=0.75,
        terminal_bonus=2.0,
        use_clip_reward=True,
    )

    def make_env():
        env = LegoVoxelRepairEnv(config=config)
        env = Monitor(env, args.log_dir)
        env = ActionMasker(env, mask_fn)
        return env

    env = DummyVecEnv([make_env])
    policy_kwargs = dict(
        features_extractor_class=Lego3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskablePPO(
        MaskableActorCriticCnnPolicy,
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        device=device,
        max_grad_norm=0.5,
    )

    with torch.no_grad():
        if hasattr(model.policy, "action_net"):
            model.policy.action_net.weight.data *= 0.01
            model.policy.action_net.bias.data.fill_(0.0)

    print(f"[Train] Device: {model.device}")
    print(f"[Train] Timesteps: {args.total_timesteps}")
    print(f"[Train] Risk checkpoint: {args.risk_checkpoint}")

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=LegoMetricsCallback())
    except KeyboardInterrupt:
        print("[Train] Interrupted by user.")

    save_path = os.path.join(args.log_dir, "ppo_lego_repair_final")
    model.save(save_path)
    print(f"[Train] Model saved to {save_path}.zip")


if __name__ == "__main__":
    main()
