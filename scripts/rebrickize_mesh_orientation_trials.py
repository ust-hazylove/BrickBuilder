import argparse
import json
from pathlib import Path

import numpy as np

from baseline_common import ensure_dir
from modules.brick_mapper import BrickMapper

try:
    from modules.mesh_utils import MeshUtils
except Exception:
    MeshUtils = None


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


def load_voxels(mesh_path: Path, bricks_json_path: Path | None, resolution: int) -> np.ndarray:
    if bricks_json_path is not None:
        return load_voxels_from_bricks_json(bricks_json_path)
    if MeshUtils is None:
        raise RuntimeError("MeshUtils/open3d unavailable; use --bricks_json instead of --mesh.")
    return MeshUtils.glb_to_voxels(mesh_path, resolution=resolution, fill=True)


def save_variant(mesh_path: Path | None, bricks_json_path: Path | None, output_dir: Path, variant_name: str, ops, resolution: int):
    voxels = load_voxels(mesh_path, bricks_json_path, resolution)
    transformed = apply_ops(voxels, ops)

    mapper = BrickMapper()
    structural = mapper.map_voxels_to_bricks(transformed, color_grid=None, verbose=False)
    # Qualitative baseline comparison must use only the standard brick library.
    styled = structural

    variant_dir = ensure_dir(output_dir / variant_name)
    mpd_path = variant_dir / f"{variant_name}.mpd"
    json_path = variant_dir / f"{variant_name}_bricks.json"
    vox_path = variant_dir / f"{variant_name}_voxels.npz"
    summary_path = variant_dir / "summary.json"

    write_mpd(styled, mpd_path)
    json_path.write_text(json.dumps(styled, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(vox_path, voxels=transformed.astype(np.uint8))
    summary_path.write_text(
        json.dumps(
            {
                "mesh_path": str(mesh_path) if mesh_path else None,
                "bricks_json_path": str(bricks_json_path) if bricks_json_path else None,
                "resolution": resolution,
                "variant_name": variant_name,
                "ops": ops,
                "occupied_voxels": int(transformed.sum()),
                "brick_count": len(styled),
                "mpd_path": str(mpd_path),
                "bricks_json_path": str(json_path),
                "voxels_path": str(vox_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"WRITE {mpd_path}")


def main():
    parser = argparse.ArgumentParser(description="Re-brickize a mesh with voxel-space orientation trials.")
    parser.add_argument("--mesh", default=None, help="Input mesh path (.glb/.obj supported by MeshUtils).")
    parser.add_argument("--bricks_json", default=None, help="Existing bricks.json path used to reconstruct occupancy.")
    parser.add_argument("--output_dir", required=True, help="Output directory for trial variants.")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument(
        "--variants_json",
        default=None,
        help='JSON string, e.g. {"trial_a":["rot_x_90"],"trial_b":["rot_x_270","mirror_z"]}',
    )
    parser.add_argument("--variants_file", default=None, help="Path to a JSON file that maps variant names to ops.")
    args = parser.parse_args()

    if not args.mesh and not args.bricks_json:
        raise ValueError("Provide either --mesh or --bricks_json.")
    mesh_path = Path(args.mesh).resolve() if args.mesh else None
    bricks_json_path = Path(args.bricks_json).resolve() if args.bricks_json else None
    output_dir = Path(args.output_dir).resolve()
    if args.variants_file:
        variants = json.loads(Path(args.variants_file).resolve().read_text(encoding="utf-8-sig"))
    elif args.variants_json:
        variants = json.loads(args.variants_json)
    else:
        raise ValueError("Provide either --variants_json or --variants_file.")
    for variant_name, ops in variants.items():
        save_variant(mesh_path, bricks_json_path, output_dir, variant_name, ops, args.resolution)


if __name__ == "__main__":
    main()
