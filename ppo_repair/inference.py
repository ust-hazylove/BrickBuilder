import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Env import EnvConfig, LegoVoxelRepairEnv
from train import Lego3DCNN, mask_fn


def visualize_results(initial_vox, final_vox, unstable_mask, save_path):
    """Save a three-panel visualization of the PPO repair actions."""
    added_voxels = (final_vox > 0) & (initial_vox == 0)

    fig = plt.figure(figsize=(18, 6))
    panels = [
        ("Initial state\n(red = unstable)", initial_vox > 0, unstable_mask, "red"),
        (f"Repair actions\n(added {int(added_voxels.sum())} voxels)", initial_vox > 0, added_voxels, "gold"),
        ("Final repaired state", final_vox > 0, None, "skyblue"),
    ]

    for index, (title, base, highlight, highlight_color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.set_title(title, fontsize=12)
        if base.any():
            ax.voxels(base, facecolors="lightgray", edgecolor="gray", alpha=0.12)
        if highlight is not None and highlight.any():
            ax.voxels(highlight, facecolors=highlight_color, edgecolor="black", alpha=0.85)
        elif index == 3 and base.any():
            ax.voxels(base, facecolors=highlight_color, edgecolor="navy", alpha=0.6)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved={save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO repair inference on prepared voxel samples.")
    parser.add_argument("--model_path", default="weights/ppo_lego_repair_final.zip")
    parser.add_argument("--dataset_root", default="ppo_repair/data/train_voxels_16")
    parser.add_argument("--risk_checkpoint", default="weights/high_risk_predictor_styled_best.pt")
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--output_dir", default="ppo_repair/inference_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_clip_reward", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = EnvConfig(
        dataset_root=args.dataset_root,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        risk_checkpoint=args.risk_checkpoint,
        use_risk_predictor=bool(args.risk_checkpoint),
        use_clip_reward=not args.no_clip_reward,
    )
    raw_env = LegoVoxelRepairEnv(config=config)
    env = ActionMasker(raw_env, mask_fn)

    model = MaskablePPO.load(
        str(model_path),
        custom_objects={"features_extractor_class": Lego3DCNN},
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    for episode in range(1, args.episodes + 1):
        print(f"\n--- PPO repair episode {episode} ---")
        obs, _ = env.reset()
        initial_vox = env.unwrapped.current_vox.copy()
        initial_unstable_mask = env.unwrapped._get_unstable_mask().copy()

        done = False
        total_reward = 0.0
        steps = 0
        info = {}
        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)

        final_vox = env.unwrapped.current_vox.copy()
        final_unstable_count = int(info.get("unstable_count", 0))
        print(f"steps={steps} reward={total_reward:.3f} final_unstable={final_unstable_count}")
        visualize_results(
            initial_vox,
            final_vox,
            initial_unstable_mask,
            output_dir / f"inference_result_{episode:02d}.png",
        )


if __name__ == "__main__":
    main()
