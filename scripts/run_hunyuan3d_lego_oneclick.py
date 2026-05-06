import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="One-command runner for Hunyuan3D LEGO LoRA finetuning.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--hunyuan_repo", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--epochs", type=str, default="10")
    parser.add_argument("--batch_size", type=str, default="1")
    parser.add_argument("--grad_accum_steps", type=str, default="4")
    parser.add_argument("--learning_rate", type=str, default="1e-4")
    parser.add_argument("--surface_points", type=str, default="8192")
    parser.add_argument("--lora_rank", type=str, default="16")
    parser.add_argument("--lora_alpha", type=str, default="32")
    parser.add_argument("--cache_latents", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    verify_cmd = [
        sys.executable,
        str(script_dir / "verify_lego_style_dataset.py"),
        "--dataset_root",
        args.dataset_root,
        "--check_samples",
        "16",
    ]
    subprocess.run(verify_cmd, check=True)

    train_cmd = [
        sys.executable,
        str(script_dir / "train_hunyuan3d_lego_lora.py"),
        "--dataset_root",
        args.dataset_root,
        "--hunyuan_repo",
        args.hunyuan_repo,
        "--model_path",
        args.model_path,
        "--subfolder",
        args.subfolder,
        "--output_dir",
        args.output_dir,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--epochs",
        args.epochs,
        "--batch_size",
        args.batch_size,
        "--grad_accum_steps",
        args.grad_accum_steps,
        "--learning_rate",
        args.learning_rate,
        "--surface_points",
        args.surface_points,
        "--lora_rank",
        args.lora_rank,
        "--lora_alpha",
        args.lora_alpha,
    ]
    if args.cache_latents:
        train_cmd.append("--cache_latents")

    subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
