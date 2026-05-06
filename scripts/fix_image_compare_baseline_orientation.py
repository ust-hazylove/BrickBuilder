import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TYPE1_PREFIX = "1 "

# Transform matrices are applied as:
# p' = T @ (p - center) + center
# R' = T @ R
TRANSFORMS = {
    "triposr": (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    "instantmesh": (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    ),
}


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


def transform_pos(pos: Tuple[float, float, float], center: Tuple[float, float, float], mat) -> Tuple[float, float, float]:
    shifted = tuple(pos[i] - center[i] for i in range(3))
    rotated = mat_vec_mul(mat, shifted)
    return tuple(rotated[i] + center[i] for i in range(3))


def transform_rot(rot_flat: Iterable[float], mat) -> List[float]:
    rot = tuple(tuple(rot_flat[r * 3 + c] for c in range(3)) for r in range(3))
    return flatten_mat(mat_mul(mat, rot))


def fix_mpd_file(path: Path, method: str) -> Tuple[float, float, float]:
    transform = TRANSFORMS[method]
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    center = compute_center_from_mpd(lines)

    out_lines: List[str] = []
    for line in lines:
        if line.startswith("0 ORIENTATION_FIXED "):
            continue
        entry = parse_type1(line)
        if entry is None:
            out_lines.append(line if line.endswith("\n") else line + "\n")
            continue
        entry["pos"] = transform_pos(entry["pos"], center, transform)
        entry["rot"] = transform_rot(entry["rot"], transform)
        out_lines.append(format_type1(entry))
    path.write_text("".join(out_lines), encoding="utf-8")
    return center


def fix_bricks_json(path: Path, method: str, center: Tuple[float, float, float]) -> None:
    transform = TRANSFORMS[method]
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for brick in data:
        if "pos" in brick and isinstance(brick["pos"], list) and len(brick["pos"]) == 3:
            brick["pos"] = list(transform_pos(tuple(brick["pos"]), center, transform))
            changed = True
        if "rot" in brick and isinstance(brick["rot"], list) and len(brick["rot"]) == 9:
            brick["rot"] = transform_rot(brick["rot"], transform)
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


def fix_structure_json(path: Path, method: str, center: Tuple[float, float, float]) -> None:
    transform = TRANSFORMS[method]
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return
    changed = False
    for node in nodes:
        if "pos" in node and isinstance(node["pos"], list) and len(node["pos"]) == 3:
            node["pos"] = list(transform_pos(tuple(node["pos"]), center, transform))
            changed = True
    if changed:
        recompute_layers(nodes)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_target_cases(root: Path) -> Iterable[Tuple[str, Path]]:
    for method in ("instantmesh", "triposr"):
        for mpd_path in sorted(root.rglob(f"{method}.mpd")):
            yield method, mpd_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix image_compare baseline brick orientation for instantmesh/triposr.")
    parser.add_argument("--input_root", required=True)
    args = parser.parse_args()

    root = Path(args.input_root).resolve()
    fixed = 0
    for method, mpd_path in iter_target_cases(root):
        center = fix_mpd_file(mpd_path, method)

        bricks_path = mpd_path.with_name(f"{method}_bricks.json")
        if bricks_path.exists():
            fix_bricks_json(bricks_path, method, center)

        structure_path = mpd_path.with_name(f"{method}_structure.json")
        if structure_path.exists():
            fix_structure_json(structure_path, method, center)

        fixed += 1
        print(f"FIXED {method} {mpd_path}")

    print(f"TOTAL_FIXED={fixed}")


if __name__ == "__main__":
    main()
