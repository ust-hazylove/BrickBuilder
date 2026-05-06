import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from modules.brick_mapper import BrickMapper
from modules.mesh_utils import MeshUtils
from ppo_repair.ldr_dataset import load_ldr_as_voxels, parse_ldr_file


DEFAULT_IMAGE_ROOTS = [
    Path("image_inputs"),
    Path("data/source_images"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a LEGO-style dataset for Hunyuan3D geometry finetuning.")
    parser.add_argument("--ldr_root", type=str, default="data/source_ldr")
    parser.add_argument(
        "--output_root",
        type=str,
        default="output/lego_style_dataset_v1",
    )
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--image_roots", nargs="*", default=[str(p) for p in DEFAULT_IMAGE_ROOTS])
    return parser.parse_args()


def slug_to_prompt(stem: str) -> str:
    return f"a lego-style {extract_category(stem).replace('_', ' ')}".strip()


def extract_category(stem: str) -> str:
    cleaned = stem
    for suffix in ("_rotate_x", "_rotate_y", "_rotate_z"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    parts = cleaned.split("_")
    category_parts: List[str] = []
    for token in parts:
        if token.isdigit():
            break
        if token.startswith("s") and len(token) >= 3 and token[1:].isdigit():
            break
        category_parts.append(token)
    if not category_parts:
        return cleaned
    return "_".join(category_parts)


def candidate_image_names(stem: str) -> List[str]:
    candidates = [stem]
    if stem.endswith("_rotate_x"):
        candidates.append(stem[:-9])
    if "_s" in stem:
        prefix = stem.split("_s")[0]
        candidates.append(prefix)
        candidates.append(prefix + "_rotate_x")
    deduped = []
    seen = set()
    for name in candidates:
        if name and name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def find_best_image(stem: str, image_roots: Sequence[Path]) -> Tuple[Optional[Path], str]:
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for base in candidate_image_names(stem):
        for root in image_roots:
            if not root.exists():
                continue
            for ext in exts:
                candidate = root / f"{base}{ext}"
                if candidate.exists():
                    return candidate, root.name
    return None, "missing"


def copy_image_as_png(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.save(dst, format="PNG")


def render_bricks(brick_list: List[dict], save_path: Path, title: str):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f5f0e6")
    ax.set_facecolor("#f5f0e6")

    for brick in brick_list:
        x, y, z = brick["grid_pos"]
        dx, dy = brick["size"]
        kind = brick.get("type_name", "")
        dz = 0.35 if "tile" in kind else 1.0
        if "slope" in kind:
            color = "#d0a15f"
        elif "tile" in kind:
            color = "#efe9df"
        else:
            color = "#c6472d"

        ax.bar3d(
            x,
            y,
            z,
            dx,
            dy,
            dz,
            color=color,
            edgecolor="#2e251d",
            linewidth=0.3,
            shade=True,
            alpha=0.98,
        )

    max_x = max((brick["grid_pos"][0] + brick["size"][0] for brick in brick_list), default=1)
    max_y = max((brick["grid_pos"][1] + brick["size"][1] for brick in brick_list), default=1)
    max_z = max((brick["grid_pos"][2] + 1 for brick in brick_list), default=1)
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_zlim(0, max_z)
    ax.set_box_aspect((max(max_x, 1), max(max_y, 1), max(max_z * 1.2, 1)))
    ax.view_init(elev=25, azim=-55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_voxel_triptych(voxels: np.ndarray, save_path: Path):
    occ = voxels > 0
    top = occ.max(axis=2).astype(np.uint8) * 255
    front = occ.max(axis=1).astype(np.uint8) * 255
    side = occ.max(axis=0).astype(np.uint8) * 255
    canvas = np.full((occ.shape[0], occ.shape[1] * 3 + 8, 3), 248, dtype=np.uint8)
    canvas[:, : occ.shape[1], :] = top[:, :, None]
    canvas[:, occ.shape[1] + 4 : occ.shape[1] * 2 + 4, :] = front[:, :, None]
    canvas[:, occ.shape[1] * 2 + 8 :, :] = side[:, :, None]
    Image.fromarray(canvas).save(save_path)


def save_brick_json(bricks: Sequence[dict], save_path: Path):
    serializable = []
    for brick in bricks:
        item = dict(brick)
        for key in ("pos", "grid_pos", "size", "rot"):
            if key in item:
                item[key] = list(item[key])
        serializable.append(item)
    save_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def split_items(items: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    random.Random(seed).shuffle(items)
    split_idx = int(math.floor(len(items) * train_ratio))
    return items[:split_idx], items[split_idx:]


def build_sample(
    ldr_path: Path,
    output_root: Path,
    image_roots: Sequence[Path],
    mapper: BrickMapper,
    grid_size: int,
) -> Dict[str, object]:
    stem = ldr_path.stem
    sample_dir = output_root / "samples" / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    voxels = load_ldr_as_voxels(str(ldr_path), grid_size=grid_size)
    color_grid = np.full(voxels.shape + (3,), 255, dtype=np.uint8)
    structural = mapper.map_voxels_to_bricks(voxels, color_grid, verbose=False)
    styled = mapper.apply_surface_finishing(structural, voxels)
    parsed_bricks = parse_ldr_file(str(ldr_path))

    input_image_path, image_source = find_best_image(stem, image_roots)
    copied_input = None
    if input_image_path is not None:
        copied_input = sample_dir / "input.png"
        copy_image_as_png(input_image_path, copied_input)

    target_vox_path = sample_dir / "target_voxels.npz"
    np.savez_compressed(target_vox_path, voxels=voxels.astype(np.uint8))

    target_mesh_path = sample_dir / "target_mesh.glb"
    MeshUtils.save_voxels_as_mesh(voxels.astype(bool), str(target_mesh_path))

    render_path = sample_dir / "target_render.png"
    render_bricks(styled, render_path, title=f"{stem} | lego target")

    triptych_path = sample_dir / "target_triptych.png"
    save_voxel_triptych(voxels.astype(np.uint8), triptych_path)

    brick_json_path = sample_dir / "target_bricks.json"
    save_brick_json(styled, brick_json_path)

    shutil.copy2(ldr_path, sample_dir / "target.ldr")

    category = extract_category(stem)
    prompt = slug_to_prompt(stem)
    meta = {
        "id": stem,
        "prompt": prompt,
        "category": category,
        "source_ldr": str(ldr_path),
        "input_image": str(copied_input) if copied_input is not None else None,
        "input_image_source": image_source,
        "target_voxels": str(target_vox_path),
        "target_mesh": str(target_mesh_path),
        "target_render": str(render_path),
        "target_triptych": str(triptych_path),
        "target_bricks": str(brick_json_path),
        "grid_size": grid_size,
        "voxel_count": int(voxels.sum()),
        "brick_count": len(styled),
        "parsed_ldr_brick_count": len(parsed_bricks),
        "slope_count": sum(1 for brick in styled if "Slope" in brick.get("name", "")),
        "tile_count": sum(1 for brick in styled if "Tile" in brick.get("name", "")),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def iter_ldr_files(ldr_root: Path, limit: Optional[int]) -> Iterable[Path]:
    files = sorted(ldr_root.glob("*.ldr"))
    if limit is not None:
        files = files[:limit]
    return files


def main():
    args = parse_args()
    random.seed(args.seed)

    ldr_root = Path(args.ldr_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_roots = [Path(path) for path in args.image_roots]

    mapper = BrickMapper()
    manifest: List[Dict[str, object]] = []
    for index, ldr_path in enumerate(iter_ldr_files(ldr_root, args.limit), start=1):
        print(f"[Dataset] {index}: {ldr_path.name}")
        try:
            manifest.append(build_sample(ldr_path, output_root, image_roots, mapper, args.grid_size))
        except Exception as exc:
            print(f"[Dataset] Failed on {ldr_path.name}: {exc}")

    ids = [row["id"] for row in manifest]
    train_ids, val_ids = split_items(ids, args.train_ratio, args.seed)
    train_set = set(train_ids)

    for row in manifest:
        row["split"] = "train" if row["id"] in train_set else "val"

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_root / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
    (output_root / "val.txt").write_text("\n".join(val_ids), encoding="utf-8")

    with_images = sum(1 for row in manifest if row.get("input_image"))
    summary = {
        "samples": len(manifest),
        "grid_size": args.grid_size,
        "with_input_images": with_images,
        "missing_input_images": len(manifest) - with_images,
        "avg_voxels": float(np.mean([row["voxel_count"] for row in manifest])) if manifest else 0.0,
        "avg_bricks": float(np.mean([row["brick_count"] for row in manifest])) if manifest else 0.0,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "image_roots": [str(path) for path in image_roots],
        "manifest": str(manifest_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
