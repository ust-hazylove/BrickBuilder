import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.brick_mapper import BrickMapper
from modules.mesh_utils import MeshUtils
import modules.voxel_to_bricks as v2b


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TripoSG image baseline GLBs into standard-brick MPDs.")
    parser.add_argument(
        "--image_compare_root",
        type=Path,
        default=Path(r"qualitative_pack_100cases_20260417\image_compare"),
    )
    parser.add_argument("--case", type=str, default="", help="Optional single case folder name.")
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--ops", type=str, default="identity", help="Comma-separated voxel orientation ops.")
    parser.add_argument(
        "--rotation-basis",
        choices=["mapper_default", "studs_up"],
        default="mapper_default",
        help="mapper_default keeps BrickMapper rotations; studs_up removes the extra X flip so visible studs face up in LDraw renders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_mpd(bricks, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("0 FILE MAIN.ldr\n")
        handle.write("0 Name: MAIN.ldr\n")
        handle.write("0 Author: Img2Build Auto\n")
        for brick in bricks:
            a, b, c, d, e, g, h, i, j = brick["rot"]
            x, y, z = brick["pos"]
            handle.write(
                f"1 {brick['color']} {x:.3f} {y:.3f} {z:.3f} "
                f"{a} {b} {c} {d} {e} {g} {h} {i} {j} {brick['file']}\n"
            )
            handle.write("0 STEP\n")
        handle.write("0 NOFILE\n")


def apply_ops(voxels: np.ndarray, ops: list[str]) -> np.ndarray:
    out = voxels
    for op in ops:
        if op in {"", "identity"}:
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


def apply_rotation_basis(bricks, rotation_basis: str):
    if rotation_basis == "mapper_default":
        return bricks
    if rotation_basis != "studs_up":
        raise ValueError(f"Unsupported rotation basis: {rotation_basis}")

    updated = []
    for brick in bricks:
        copy = dict(brick)
        copy["rot"] = list(v2b.rotz(int(copy.get("ori_quarter", 0)) * 90))
        updated.append(copy)
    return updated


def process_case(case_dir: Path, resolution: int, ops: list[str], rotation_basis: str, overwrite: bool) -> bool:
    mesh_path = case_dir / "triposg.glb"
    if not mesh_path.exists():
        return False

    mpd_path = case_dir / "triposg.mpd"
    if mpd_path.exists() and not overwrite:
        print(f"SKIP existing {mpd_path}")
        return False

    voxels = MeshUtils.glb_to_voxels(mesh_path, resolution=resolution, fill=True)
    transformed = apply_ops(voxels, ops)

    mapper = BrickMapper()
    bricks = mapper.map_voxels_to_bricks(transformed, color_grid=None, verbose=False)
    bricks = apply_rotation_basis(bricks, rotation_basis)
    structure = make_structure(bricks)

    write_mpd(bricks, mpd_path)
    (case_dir / "triposg_bricks.json").write_text(json.dumps(bricks, indent=2, ensure_ascii=False), encoding="utf-8")
    (case_dir / "triposg_structure.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "method": "triposg",
        "source_mesh": str(mesh_path),
        "resolution": resolution,
        "orientation_ops": ops,
        "rotation_basis": rotation_basis,
        "occupied_voxels": int(transformed.sum()),
        "brick_count": len(bricks),
        "standard_bricks_only": True,
        "mpd_path": str(mpd_path),
    }
    (case_dir / "triposg_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"UPDATED {case_dir.name}: voxels={int(transformed.sum())}, bricks={len(bricks)}")
    return True


def main():
    args = parse_args()
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    if args.case:
        case_dirs = [args.image_compare_root / args.case]
    else:
        case_dirs = sorted(path for path in args.image_compare_root.iterdir() if path.is_dir())

    total = 0
    for case_dir in case_dirs:
        if process_case(case_dir, args.resolution, ops, args.rotation_basis, args.overwrite):
            total += 1
    print(f"TOTAL_UPDATED={total}")


if __name__ == "__main__":
    main()
