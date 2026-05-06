import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.brick_mapper import BrickMapper


METHOD_OPS = {
    "instantmesh": ["rot_x_90"],
    "triposr": ["rot_x_90", "mirror_z"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild image_compare baselines with corrected voxel orientation and standard-brick-only mapping."
    )
    parser.add_argument("--input_root", required=True, help="Root image_compare directory.")
    parser.add_argument("--backup_dirname", default="_baseline_backup_before_rebrick", help="Per-case backup folder name.")
    return parser.parse_args()


def write_mpd(bricks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("0 FILE MAIN.ldr\n")
        f.write("0 Name: MAIN.ldr\n")
        f.write("0 Author: Img2Build Auto\n")
        for brick in bricks:
            a, b, c, d, e, g, h, i, j = brick["rot"]
            x, y, z = brick["pos"]
            f.write(
                f"1 {brick['color']} {x:.3f} {y:.3f} {z:.3f} "
                f"{a} {b} {c} {d} {e} {g} {h} {i} {j} {brick['file']}\n"
            )
            f.write("0 STEP\n")
        f.write("0 NOFILE\n")


def apply_ops(voxels: np.ndarray, ops):
    out = voxels
    for op in ops:
        if op == "identity":
            continue
        if op == "rot_x_90":
            out = np.rot90(out, k=1, axes=(1, 2))
        elif op == "rot_x_180":
            out = np.rot90(out, k=2, axes=(1, 2))
        elif op == "rot_x_270":
            out = np.rot90(out, k=3, axes=(1, 2))
        elif op == "rot_y_90":
            out = np.rot90(out, k=1, axes=(0, 2))
        elif op == "rot_y_180":
            out = np.rot90(out, k=2, axes=(0, 2))
        elif op == "rot_y_270":
            out = np.rot90(out, k=3, axes=(0, 2))
        elif op == "rot_z_90":
            out = np.rot90(out, k=1, axes=(0, 1))
        elif op == "rot_z_180":
            out = np.rot90(out, k=2, axes=(0, 1))
        elif op == "rot_z_270":
            out = np.rot90(out, k=3, axes=(0, 1))
        elif op == "mirror_x":
            out = np.flip(out, axis=0)
        elif op == "mirror_y":
            out = np.flip(out, axis=1)
        elif op == "mirror_z":
            out = np.flip(out, axis=2)
        else:
            raise ValueError(f"Unsupported op: {op}")
    return np.ascontiguousarray(out)


def load_voxels_from_bricks_json(bricks_json_path: Path) -> np.ndarray:
    bricks = json.loads(bricks_json_path.read_text(encoding="utf-8"))
    max_x = max(int(b["grid_pos"][0]) + int(b["size"][0]) for b in bricks)
    max_y = max(int(b["grid_pos"][1]) + int(b["size"][1]) for b in bricks)
    max_z = max(int(b["grid_pos"][2]) + 1 for b in bricks)
    voxels = np.zeros((max_x, max_y, max_z), dtype=bool)
    for brick in bricks:
        x, y, z = [int(v) for v in brick["grid_pos"]]
        dx, dy = [int(v) for v in brick["size"]]
        voxels[x:x + dx, y:y + dy, z] = True
    return voxels


def make_structure(bricks):
    heights = sorted({brick["pos"][1] for brick in bricks}, reverse=True)
    layer_map = {height: idx for idx, height in enumerate(heights)}
    nodes = []
    for brick in bricks:
        nodes.append(
            {
                "id": brick["id"],
                "file": brick["file"],
                "type_name": brick.get("type_name"),
                "struct_type": brick.get("struct_type"),
                "grid_pos": brick["grid_pos"],
                "size": brick["size"],
                "pos": brick["pos"],
                "layer": layer_map[brick["pos"][1]],
                "lateral_neighbors": [],
            }
        )
    return {"nodes": nodes, "edges": []}


def backup_file(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    dst = backup_root / path.name
    if not dst.exists():
        shutil.copy2(path, dst)


def process_method(case_dir: Path, method: str, ops, backup_dirname: str):
    bricks_json_path = case_dir / f"{method}_bricks.json"
    if not bricks_json_path.exists():
        return False

    backup_root = case_dir / backup_dirname
    for path in [
        case_dir / f"{method}.mpd",
        case_dir / f"{method}_bricks.json",
        case_dir / f"{method}_structure.json",
        case_dir / f"{method}_summary.json",
        case_dir / f"{method}_reference_rgba.png",
        case_dir / f"{method}_white_render_hd.png",
        case_dir / f"{method}_white_render.png",
    ]:
        backup_file(path, backup_root)

    voxels = load_voxels_from_bricks_json(bricks_json_path)
    transformed = apply_ops(voxels, ops)
    mapper = BrickMapper()
    bricks = mapper.map_voxels_to_bricks(transformed, color_grid=None, verbose=False)
    structure = make_structure(bricks)

    mpd_path = case_dir / f"{method}.mpd"
    structure_path = case_dir / f"{method}_structure.json"
    summary_path = case_dir / f"{method}_summary.json"

    write_mpd(bricks, mpd_path)
    bricks_json_path.write_text(json.dumps(bricks, indent=2, ensure_ascii=False), encoding="utf-8")
    structure_path.write_text(json.dumps(structure, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "method": method,
        "orientation_ops": ops,
        "occupied_voxels": int(transformed.sum()),
        "brick_count": len(bricks),
        "standard_bricks_only": True,
        "mpd_path": str(mpd_path),
        "bricks_json_path": str(bricks_json_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    case_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()])
    total = 0
    for case_dir in case_dirs:
        for method, ops in METHOD_OPS.items():
            if process_method(case_dir, method, ops, args.backup_dirname):
                total += 1
                print(f"UPDATED {case_dir.name} {method}")
    print(f"TOTAL_UPDATED={total}")


if __name__ == "__main__":
    main()
