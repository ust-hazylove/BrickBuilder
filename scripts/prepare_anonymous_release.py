"""Build a clean anonymous-release staging folder.

The script copies reviewer-facing source code, docs, sample inputs, and small model
checkpoints while excluding generated outputs, logs, paper build products, local
paths, and nested git metadata.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements_hunyuan3d_lego_finetune.txt",
    "environments.yaml",
    "app.py",
    "core_pipeline.py",
    "all_case_metrics.csv",
    "benchmark_batch.py",
    "comprehensive_benchmark.py",
    "fix_ldr_orientation.py",
    "mapping_vis.py",
]

REQUIRED_DIRS = [
    "modules",
    "weights",
    "image_inputs",
]

DOC_FILES = [
    "docs/REVIEWER_GUIDE.md",
    "docs/ANONYMOUS_RELEASE_CHECKLIST.md",
    "docs/text_comparison_prompts_v1.csv",
]

FILTERED_DIRS = {
    "scripts": ["__pycache__", "*.pyc"],
    "ppo_repair": ["__pycache__", "*.pyc", "*.png", "data", "logs", "inference_outputs"],
    "assembly_sequence": [
        "__pycache__",
        "*.pyc",
        "*.png",
        "*.mp4",
        "out",
        "output_video",
        "model",
        "dataset",
        "images",
        ".git",
    ],
    "Experiments": [
        "__pycache__",
        "*.pyc",
        "*.png",
        "*.pdf",
        "benchmark_data",
        "temp_renders",
        "smart_demo_v2_output",
        "voxel_data",
        "weights",
    ],
}

OPTIONAL_HUNYUAN_IGNORE = [".git", "__pycache__", "*.pyc", "hy3dgen.egg-info"]

GLOBAL_IGNORES = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
]


def should_ignore(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def make_ignore(patterns: Sequence[str]):
    all_patterns = list(GLOBAL_IGNORES) + list(patterns)

    def _ignore(_dir: str, names: List[str]) -> set[str]:
        return {name for name in names if should_ignore(name, all_patterns)}

    return _ignore


def copy_file(src_rel: str, out_root: Path, dry_run: bool, manifest: List[str]) -> None:
    src = ROOT / src_rel
    dst = out_root / src_rel
    if not src.exists():
        manifest.append(f"missing file: {src_rel}")
        return
    manifest.append(f"copy file: {src_rel}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_dir(src_rel: str, out_root: Path, dry_run: bool, manifest: List[str], ignores: Iterable[str] = ()) -> None:
    src = ROOT / src_rel
    dst = out_root / src_rel
    if not src.exists():
        manifest.append(f"missing dir: {src_rel}")
        return
    manifest.append(f"copy dir: {src_rel}")
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=make_ignore(list(ignores)))


def write_manifest(out_root: Path, manifest: Sequence[str], dry_run: bool) -> None:
    text = "\n".join(manifest) + "\n"
    print(text)
    if not dry_run:
        (out_root / "RELEASE_MANIFEST.txt").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an anonymous reviewer-facing release folder.")
    parser.add_argument("--output", default="anonymous_release", help="Output staging directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned copies without writing files.")
    parser.add_argument("--force", action="store_true", help="Replace the output directory if it already exists.")
    parser.add_argument(
        "--include-hunyuan-vendor",
        action="store_true",
        help="Also copy the vendored Hunyuan3D-2 folder without nested git metadata.",
    )
    parser.add_argument(
        "--include-ppo-training-data",
        action="store_true",
        help="Also copy ppo_repair/data for PPO training reproduction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = (ROOT / args.output).resolve()
    manifest: List[str] = []

    if out_root == ROOT:
        raise ValueError("Output directory must not be the repository root.")

    if out_root.exists() and not args.dry_run:
        if not args.force:
            raise FileExistsError(f"Output exists: {out_root}. Use --force to replace it.")
        shutil.rmtree(out_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    for rel in REQUIRED_FILES:
        copy_file(rel, out_root, args.dry_run, manifest)
    for rel in DOC_FILES:
        copy_file(rel, out_root, args.dry_run, manifest)
    for rel in REQUIRED_DIRS:
        copy_dir(rel, out_root, args.dry_run, manifest)
    for rel, ignores in FILTERED_DIRS.items():
        copy_dir(rel, out_root, args.dry_run, manifest, ignores)

    if args.include_hunyuan_vendor:
        copy_dir("Hunyuan3D-2", out_root, args.dry_run, manifest, OPTIONAL_HUNYUAN_IGNORE)
    else:
        manifest.append("skip optional dir: Hunyuan3D-2 (use --include-hunyuan-vendor)")

    if args.include_ppo_training_data:
        copy_dir("ppo_repair/data", out_root, args.dry_run, manifest)
    else:
        manifest.append("skip optional dir: ppo_repair/data (use --include-ppo-training-data)")

    write_manifest(out_root, manifest, args.dry_run)


if __name__ == "__main__":
    main()
