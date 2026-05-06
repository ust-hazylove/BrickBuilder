import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.brick_mapper import BrickMapper
from modules.mesh_utils import MeshUtils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild text_compare trellis_text_large from mesh using identity voxel orientation."
    )
    parser.add_argument("--input_root", required=True, help="Root text_compare directory.")
    parser.add_argument("--backup_dirname", default="_trellis_backup_before_mesh_rebrick", help="Per-case backup folder name.")
    parser.add_argument("--resolution", type=int, default=16, help="Voxel resolution.")
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


def process_case(case_dir: Path, backup_dirname: str, resolution: int):
    method = "trellis_text_large"
    mesh_path = case_dir / f"{method}_mesh.glb"
    if not mesh_path.exists():
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

    voxels = MeshUtils.glb_to_voxels(mesh_path, resolution=resolution, fill=True)
    mapper = BrickMapper()
    bricks = mapper.map_voxels_to_bricks(voxels, color_grid=None, verbose=False)
    structure = make_structure(bricks)

    mpd_path = case_dir / f"{method}.mpd"
    bricks_json_path = case_dir / f"{method}_bricks.json"
    structure_path = case_dir / f"{method}_structure.json"
    summary_path = case_dir / f"{method}_summary.json"

    write_mpd(bricks, mpd_path)
    bricks_json_path.write_text(json.dumps(bricks, indent=2, ensure_ascii=False), encoding="utf-8")
    structure_path.write_text(json.dumps(structure, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary["mesh_rebrick"] = {
        "source": str(mesh_path),
        "resolution": resolution,
        "voxel_orientation": "identity",
        "render_rotation": ["rotate_x_180", "rotate_z_180"],
        "occupied_voxels": int(voxels.sum()),
        "brick_count": len(bricks),
        "standard_bricks_only": True,
    }
    summary["styled_brick_count"] = len(bricks)
    summary["structural_brick_count"] = len(bricks)
    summary["mpd_path"] = str(mpd_path)
    summary["styled_bricks_json"] = str(bricks_json_path)
    summary["structure_dag_path"] = str(structure_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    total = 0
    for case_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if process_case(case_dir, args.backup_dirname, args.resolution):
            total += 1
            print(f"UPDATED {case_dir.name}")
    print(f"TOTAL_UPDATED={total}")


if __name__ == "__main__":
    main()
