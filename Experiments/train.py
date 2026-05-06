import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 引入你的环境和配置
from Env import LegoVoxelRepairEnv, EnvConfig

# ==============================================================================
# 1. 自定义 Callback (记录新版指标)
# ==============================================================================
class LegoMetricsCallback(BaseCallback):
    """
    记录诊断-修复任务的关键指标到 TensorBoard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # SB3 的 locals['infos'] 是一个列表 (针对每个 env)
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # 记录实时指标
            if "unstable_count" in info:
                self.logger.record("custom/unstable_count", info["unstable_count"])
            if "added_voxels" in info:
                self.logger.record("custom/added_voxels", info["added_voxels"])
            if "delta_stable" in info:
                self.logger.record("custom/delta_stable", info["delta_stable"])
            if "valid_action" in info:
                self.logger.record("custom/valid_action_rate", info["valid_action"])
                
        return True

# ==============================================================================
# 2. 自定义 3D CNN (带正交初始化)
# ==============================================================================
class Lego3DCNN(BaseFeaturesExtractor):
    """
    3D CNN 特征提取器，针对 32x32x32 体素输入优化。
    包含 Orthogonal Initialization 以防止训练初期数值不稳定。
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # Should be 2 (Current + Mask)

        self.cnn = nn.Sequential(
            # [2, 32, 32, 32] -> [32, 16, 16, 16]
            nn.Conv3d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # [32, 16, 16, 16] -> [64, 8, 8, 8]
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # [64, 8, 8, 8] -> [128, 4, 4, 4]
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
        )

        # --- 权重初始化 (关键修复) ---
        for module in self.cnn.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    init.constant_(module.bias, 0)

        # 动态计算维度
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.LayerNorm(n_flatten),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
        # Linear 层初始化
        for module in self.modules():
             if isinstance(module, (nn.Conv3d, nn.Linear)):
                init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ==============================================================================
# 3. 辅助函数
# ==============================================================================
def mask_fn(env: gym.Env) -> np.ndarray:
    """Wrapper 穿透调用底层 env 的 action_masks"""
    return env.get_wrapper_attr("action_masks")()

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for Lego Repair")
    # 建议默认步数设大一点，以便看到收敛
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset_root", type=str, default="./data/train_voxels", help="Path to voxel data")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Log directory")
    return parser.parse_args()

# ==============================================================================
# 4. 主函数
# ==============================================================================
def main():
    args = parse_args()
    
    # 1. 强制设置 float32 (防止 FP16 下溢)
    torch.set_default_dtype(torch.float32)
    
    # 2. 初始化配置
    config = EnvConfig(
        dataset_root=args.dataset_root,
        grid_size=32,
        max_steps=32,
        # 你可以在这里调整奖励权重
        stability_reward=2.0,
        iou_penalty=0.1,
        step_penalty=-0.01
    )

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # 3. 创建环境工厂函数
    def make_env():
        env = LegoVoxelRepairEnv(config=config)
        env = Monitor(env, args.log_dir) # 记录 Episode Reward
        env = ActionMasker(env, mask_fn) # 启用 Action Masking
        return env

    env = DummyVecEnv([make_env])

    # 4. 定义 PPO 参数
    policy_kwargs = dict(
        features_extractor_class=Lego3DCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # 5. 初始化 MaskablePPO
    model = MaskablePPO(
        MaskableActorCriticCnnPolicy, # 使用 Maskable 专用 Policy 类
        env, 
        verbose=1, 
        tensorboard_log=args.log_dir,
        learning_rate=1e-4,    # 降低学习率以稳定训练
        n_steps=2048,
        batch_size=64,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        device='cuda',         # 确保使用 GPU
        max_grad_norm=0.5      # 梯度裁剪 (防止数值爆炸)
    )

    with torch.no_grad():
        if hasattr(model.policy, 'action_net'):
            model.policy.action_net.weight.data *= 0.01
            model.policy.action_net.bias.data.fill_(0.0)
            print("Fixed: Scaled down action_net weights and biases.")

    print(f"Start Training on {model.device}...")
    print(f"Total Timesteps: {args.total_timesteps}")
    
    # 6. 开始训练
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=LegoMetricsCallback()
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # 7. 保存模型
    save_path = os.path.join(args.log_dir, "ppo_lego_repair_final")
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    main()