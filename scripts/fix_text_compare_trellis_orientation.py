import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TRANSFORM = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)

RECOVER_TRANSFORM = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fix trellis_text_large orientation in text_compare.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--backup_dirname", default="_trellis_backup_before_fix")
    return parser.parse_args()


def mat_vec_mul(mat: Tuple[Tuple[float, ...], ...], vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(
        mat[row][0] * vec[0] + mat[row][1] * vec[1] + mat[row][2] * vec[2]
        for row in range(3)
    )


def mat_mul(a: Tuple[Tuple[float, ...], ...], b: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
    out = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(sum(a[r][k] * b[k][c] for k in range(3)))
        out.append(tuple(row))
    return tuple(out)


def flatten_mat(mat: Tuple[Tuple[float, ...], ...]) -> List[float]:
    return [mat[r][c] for r in range(3) for c in range(3)]


def parse_type1(line: str) -> Optional[Dict]:
    toks = line.strip().split()
    if len(toks) < 15 or toks[0] != "1":
        return None
    return {
        "color": toks[1],
        "pos": tuple(float(v) for v in toks[2:5]),
        "rot": tuple(float(v) for v in toks[5:14]),
        "part": " ".join(toks[14:]),
    }


def format_type1(entry: Dict) -> str:
    x, y, z = entry["pos"]
    a, b, c, d, e, f, g, h, i = entry["rot"]
    return (
        f"1 {entry['color']} {x:.6f} {y:.6f} {z:.6f} "
        f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} "
        f"{g:.6f} {h:.6f} {i:.6f} {entry['part']}\n"
    )


def compute_center_from_mpd(lines: Iterable[str]) -> Tuple[float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for line in lines:
        entry = parse_type1(line)
        if entry is None:
            continue
        x, y, z = entry["pos"]
        xs.append(x)
        ys.append(y)
        zs.append(z)
    if not xs:
        raise ValueError("No type-1 lines found in MPD.")
    return (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )


def transform_pos(pos: Tuple[float, float, float], center: Tuple[float, float, float]) -> Tuple[float, float, float]:
    shifted = tuple(pos[i] - center[i] for i in range(3))
    rotated = mat_vec_mul(TRANSFORM, shifted)
    return tuple(rotated[i] + center[i] for i in range(3))


def transform_rot(rot_flat: Iterable[float]) -> List[float]:
    rot = tuple(tuple(rot_flat[r * 3 + c] for c in range(3)) for r in range(3))
    return flatten_mat(mat_mul(TRANSFORM, rot))


def apply_transform_pos(pos: Tuple[float, float, float], center: Tuple[float, float, float], transform) -> Tuple[float, float, float]:
    shifted = tuple(pos[i] - center[i] for i in range(3))
    rotated = mat_vec_mul(transform, shifted)
    return tuple(rotated[i] + center[i] for i in range(3))


def apply_transform_rot(rot_flat: Iterable[float], transform) -> List[float]:
    rot = tuple(tuple(rot_flat[r * 3 + c] for c in range(3)) for r in range(3))
    return flatten_mat(mat_mul(transform, rot))


def backup_if_needed(path: Path, backup_root: Path):
    if not path.exists():
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    dst = backup_root / path.name
    if not dst.exists():
        shutil.copy2(path, dst)


def source_path(path: Path, backup_root: Path) -> Path:
    candidate = backup_root / path.name
    if candidate.exists():
        return candidate
    return path


def fix_mpd_file(path: Path, backup_root: Path) -> Tuple[float, float, float]:
    backup_if_needed(path, backup_root)
    src = source_path(path, backup_root)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    center = compute_center_from_mpd(lines)
    out_lines: List[str] = []
    for line in lines:
        entry = parse_type1(line)
        if entry is None:
            out_lines.append(line if line.endswith("\n") else line + "\n")
            continue
        recovered_pos = apply_transform_pos(entry["pos"], center, RECOVER_TRANSFORM)
        recovered_rot = apply_transform_rot(entry["rot"], RECOVER_TRANSFORM)
        entry["pos"] = apply_transform_pos(recovered_pos, center, TRANSFORM)
        entry["rot"] = apply_transform_rot(recovered_rot, TRANSFORM)
        out_lines.append(format_type1(entry))
    path.write_text("".join(out_lines), encoding="utf-8")
    return center


def fix_bricks_json(path: Path, backup_root: Path, center: Tuple[float, float, float]) -> None:
    if not path.exists():
        return
    backup_if_needed(path, backup_root)
    src = source_path(path, backup_root)
    data = json.loads(src.read_text(encoding="utf-8"))
    changed = False
    for brick in data:
        if "pos" in brick and isinstance(brick["pos"], list) and len(brick["pos"]) == 3:
            brick["pos"] = list(transform_pos(tuple(brick["pos"]), center))
            changed = True
        if "rot" in brick and isinstance(brick["rot"], list) and len(brick["rot"]) == 9:
            brick["rot"] = transform_rot(brick["rot"])
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def recompute_layers(nodes: List[Dict]) -> None:
    pos_nodes = [node for node in nodes if isinstance(node, dict) and isinstance(node.get("pos"), list) and len(node["pos"]) == 3]
    if not pos_nodes:
        return
    vertical_values = sorted({round(node["pos"][1], 6) for node in pos_nodes}, reverse=True)
    layer_map = {value: idx for idx, value in enumerate(vertical_values)}
    for node in pos_nodes:
        node["layer"] = layer_map[round(node["pos"][1], 6)]


def fix_structure_json(path: Path, backup_root: Path, center: Tuple[float, float, float]) -> None:
    if not path.exists():
        return
    backup_if_needed(path, backup_root)
    src = source_path(path, backup_root)
    data = json.loads(src.read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return
    changed = False
    for node in nodes:
        if "pos" in node and isinstance(node["pos"], list) and len(node["pos"]) == 3:
            node["pos"] = list(transform_pos(tuple(node["pos"]), center))
            changed = True
    if changed:
        recompute_layers(nodes)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.input_root).resolve()
    fixed = 0
    for mpd_path in sorted(root.rglob("trellis_text_large.mpd")):
        if args.backup_dirname in mpd_path.parts:
            continue
        backup_root = mpd_path.parent / args.backup_dirname
        center = fix_mpd_file(mpd_path, backup_root)
        fix_bricks_json(mpd_path.with_name("trellis_text_large_bricks.json"), backup_root, center)
        fix_structure_json(mpd_path.with_name("trellis_text_large_structure.json"), backup_root, center)
        fixed += 1
        print(f"FIXED {mpd_path}")
    print(f"TOTAL_FIXED={fixed}")


if __name__ == "__main__":
    main()
