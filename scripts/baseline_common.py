import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    value = value.strip("._-")
    return value or "case"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_case_dir(output_root: Path, baseline: str, case_id: str) -> Path:
    return ensure_dir(output_root / baseline / case_id)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(
    command: Sequence[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    stdin_text: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = None
    if env is not None:
        merged_env = dict(os.environ)
        merged_env.update(env)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle:
        with stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_handle:
            completed = subprocess.run(
                list(command),
                cwd=str(cwd),
                input=stdin_text,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=merged_env,
                check=False,
            )
    return int(completed.returncode)


def copy_first_existing(candidates: Iterable[Path], dst: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, dst)
            return dst
    return None


def brickize_mesh(
    mesh_path: Path,
    output_dir: Path,
    resolution: int = 16,
    rotate_axis: Optional[str] = None,
    rotate_k: int = 0,
) -> Dict:
    from modules.brick_mapper import BrickMapper
    from modules.mesh_utils import MeshUtils

    voxel_grid = MeshUtils.glb_to_voxels(mesh_path, resolution=resolution, fill=True)
    if rotate_axis and rotate_k:
        voxel_grid = MeshUtils.rotate_voxels(voxel_grid, axis=rotate_axis, k=rotate_k)

    mapper = BrickMapper()
    structural_bricks = mapper.map_voxels_to_bricks(voxel_grid, color_grid=None, verbose=False)
    styled_bricks = mapper.apply_surface_finishing(structural_bricks, voxel_grid)
    bom_df = mapper.generate_bom(styled_bricks)

    brick_dir = ensure_dir(output_dir / "brickized")
    preview_path = brick_dir / "assembly_preview.glb"
    bom_csv_path = brick_dir / "bom.csv"
    bricks_json_path = brick_dir / "styled_bricks.json"
    voxel_npz_path = brick_dir / "voxel_grid.npz"

    MeshUtils.save_voxels_as_mesh(voxel_grid, str(preview_path))
    bom_df.to_csv(bom_csv_path, index=False)
    bricks_json_path.write_text(json.dumps(styled_bricks, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(voxel_npz_path, voxels=voxel_grid.astype(np.uint8))

    return {
        "resolution": int(resolution),
        "occupied_voxels": int(voxel_grid.sum()),
        "structural_brick_count": int(len(structural_bricks)),
        "styled_brick_count": int(len(styled_bricks)),
        "preview_path": str(preview_path),
        "bom_csv_path": str(bom_csv_path),
        "styled_bricks_json": str(bricks_json_path),
        "voxel_grid_path": str(voxel_npz_path),
    }


def init_summary(
    baseline: str,
    case_id: str,
    input_kind: str,
    input_value: str,
    repo_root: Path,
    case_dir: Path,
) -> Dict:
    return {
        "baseline": baseline,
        "case_id": case_id,
        "input_kind": input_kind,
        "input_value": input_value,
        "repo_root": str(repo_root),
        "case_dir": str(case_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
