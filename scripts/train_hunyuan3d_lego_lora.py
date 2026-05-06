import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA finetune Hunyuan3D shape denoiser on LEGO-style targets.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--hunyuan_repo", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-1")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surface_points", type=int, default=8192)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--cache_latents", action="store_true")
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


class LegoStyleManifestDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str, limit: Optional[int] = None):
        self.dataset_root = dataset_root
        manifest_path = dataset_root / "manifest.jsonl"
        rows: List[Dict[str, object]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("split") == split:
                    rows.append(row)
        if limit is not None:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_manifest(rows: Sequence[Dict[str, object]]) -> Dict[str, List[object]]:
    keys = rows[0].keys()
    return {key: [row.get(key) for row in rows] for key in keys}


@dataclass
class TrainState:
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float


class HunyuanLegoTrainer:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(args.device)
        self.weight_dtype = resolve_dtype(args.dtype)

        repo_path = Path(args.hunyuan_repo)
        if not repo_path.exists():
            raise FileNotFoundError(f"Hunyuan repo not found: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from peft import LoraConfig, get_peft_model
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.surface_loaders import SurfaceLoader

        self.SurfaceLoader = SurfaceLoader
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            device=args.device,
            dtype=self.weight_dtype,
            use_safetensors=False,
        )

        self.pipeline.model.train()
        self.pipeline.vae.eval()
        self.pipeline.conditioner.eval()
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.conditioner.requires_grad_(False)
        self.pipeline.model.requires_grad_(False)

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["qkv", "proj", "linear1", "linear2", "in_layer", "out_layer", "lin"],
        )
        self.model = get_peft_model(self.pipeline.model, lora_config).to(self.device)
        self.model.train()

        self.surface_loader = self.SurfaceLoader(num_points=args.surface_points)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.latent_cache_dir = Path(args.latent_cache_dir) if args.latent_cache_dir else Path(args.dataset_root) / "latent_cache"
        if args.cache_latents:
            self.latent_cache_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_image_batch(self, image_paths: Sequence[str]):
        outputs = [self.pipeline.image_processor(Image.open(path).convert("RGBA")) for path in image_paths]
        batch = {key: [] for key in outputs[0].keys()}
        for item in outputs:
            for key, value in item.items():
                batch[key].append(value)
        for key, value in batch.items():
            if isinstance(value[0], torch.Tensor):
                batch[key] = torch.cat(value, dim=0).to(self.device, dtype=self.weight_dtype)
        return batch

    def _latent_cache_path(self, sample_id: str) -> Path:
        return self.latent_cache_dir / f"{sample_id}.pt"

    @torch.no_grad()
    def _encode_target_latents(self, sample_ids: Sequence[str], mesh_paths: Sequence[str]) -> torch.Tensor:
        latents = []
        for sample_id, mesh_path in zip(sample_ids, mesh_paths):
            cache_path = self._latent_cache_path(sample_id)
            if self.args.cache_latents and cache_path.exists():
                latent = torch.load(cache_path, map_location="cpu")
                latents.append(latent)
                continue

            mesh = trimesh.load(mesh_path, force="mesh", merge_primitives=True)
            surface = self.surface_loader(mesh)
            surface = surface.to(self.device, dtype=self.weight_dtype)
            latent = self.pipeline.vae.encode(surface, sample_posterior=False).detach().cpu()
            if self.args.cache_latents:
                torch.save(latent, cache_path)
            latents.append(latent)

        latent_batch = torch.cat(latents, dim=0).to(self.device, dtype=self.weight_dtype)
        return latent_batch

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        indices = torch.randint(
            low=0,
            high=len(self.pipeline.scheduler.timesteps) - 1,
            size=(batch_size,),
            device=self.device,
        )
        timesteps = self.pipeline.scheduler.timesteps.to(self.device, dtype=self.weight_dtype)[indices]
        return timesteps

    def _step_batch(self, batch: Dict[str, List[object]]) -> torch.Tensor:
        image_inputs = self._prepare_image_batch(batch["input_image"])
        images = image_inputs.pop("image")

        with torch.no_grad():
            cond = self.pipeline.conditioner(image=images, **image_inputs)
            latents = self._encode_target_latents(batch["id"], batch["target_mesh"])

        noise = torch.randn_like(latents)
        timesteps = self._sample_timesteps(latents.shape[0])
        noisy_latents = self.pipeline.scheduler.scale_noise(latents, timesteps, noise)
        model_t = timesteps / float(self.pipeline.scheduler.config.num_train_timesteps)

        pred = self.model(noisy_latents, model_t, cond)
        target = noise - latents
        loss = F.mse_loss(pred.float(), target.float())
        return loss

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in loader:
            loss = self._step_batch(batch)
            losses.append(float(loss.item()))
        self.model.train()
        return float(sum(losses) / max(len(losses), 1))

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        history = []
        global_step = 0
        best_val = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            epoch_losses: List[float] = []
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                loss = self._step_batch(batch)
                (loss / self.args.grad_accum_steps).backward()
                epoch_losses.append(float(loss.item()))

                if step % self.args.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

            if step % self.args.grad_accum_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

            train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            val_loss = self.evaluate(val_loader) if len(val_loader.dataset) > 0 else train_loss
            state = TrainState(epoch=epoch, global_step=global_step, train_loss=train_loss, val_loss=val_loss)
            history.append(asdict(state))

            self._write_history(history)
            print(json.dumps(asdict(state), ensure_ascii=False))

            if val_loss < best_val:
                best_val = val_loss
                best_dir = self.output_dir / "best_adapter"
                best_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(best_dir)

            if epoch % self.args.save_every_epochs == 0:
                ckpt_dir = self.output_dir / f"checkpoint-epoch{epoch:02d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)

        final_dir = self.output_dir / "final_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_dir)
        (self.output_dir / "train_args.json").write_text(json.dumps(vars(self.args), ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_history(self, history: List[Dict[str, object]]):
        (self.output_dir / "train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = Path(args.dataset_root)
    train_dataset = LegoStyleManifestDataset(dataset_root, split="train", limit=args.max_train_samples)
    val_dataset = LegoStyleManifestDataset(dataset_root, split="val", limit=args.max_val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_manifest,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_manifest,
    )

    trainer = HunyuanLegoTrainer(args)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
