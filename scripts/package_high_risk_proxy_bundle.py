import json
import shutil
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_requirements(path: Path) -> None:
    content = "\n".join(
        [
            "torch",
            "numpy",
            "scipy",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def write_sample_bricks(path: Path) -> None:
    sample = [
        {
            "id": 0,
            "type_name": "2x4",
            "struct_type": "2x4",
            "size": [4, 2],
            "grid_pos": [3, 5, 1],
            "ori_quarter": 0,
            "semantic_label": 0,
        },
        {
            "id": 1,
            "type_name": "tile_1x1",
            "struct_type": "1x1",
            "size": [1, 1],
            "grid_pos": [6, 6, 2],
            "ori_quarter": 0,
            "semantic_label": 0,
        },
    ]
    path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")


def write_launcher(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/run_high_risk_proxy_standalone.py" "$@"
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    output_root = PROJECT_ROOT / "output" / "high_risk_proxy_transfer_bundle"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_root = output_root / "high_risk_proxy_transfer_bundle"
    weights_dir = bundle_root / "weights"
    modules_dir = bundle_root / "modules"
    scripts_dir = bundle_root / "scripts"
    docs_dir = bundle_root / "docs"
    examples_dir = bundle_root / "examples"
    for folder in [weights_dir, modules_dir, scripts_dir, docs_dir, examples_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    copy_map = {
        PROJECT_ROOT / "weights" / "high_risk_predictor_styled_best.pt": weights_dir / "high_risk_predictor_styled_best.pt",
        PROJECT_ROOT / "modules" / "high_risk_predictor.py": modules_dir / "high_risk_predictor.py",
        PROJECT_ROOT / "modules" / "risk_analysis.py": modules_dir / "risk_analysis.py",
        PROJECT_ROOT / "scripts" / "run_high_risk_proxy_standalone.py": scripts_dir / "run_high_risk_proxy_standalone.py",
        PROJECT_ROOT / "docs" / "high_risk_proxy_transfer_guide.md": docs_dir / "high_risk_proxy_transfer_guide.md",
    }

    for src, dst in copy_map.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        shutil.copy2(src, dst)

    write_requirements(bundle_root / "requirements.txt")
    write_sample_bricks(examples_dir / "sample_bricks.json")
    write_launcher(bundle_root / "run_proxy.sh")

    zip_path = output_root / "high_risk_proxy_transfer_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_root.rglob("*"):
            zf.write(file_path, arcname=file_path.relative_to(output_root))

    print(f"Bundle directory: {bundle_root}")
    print(f"Bundle zip: {zip_path}")


if __name__ == "__main__":
    main()
