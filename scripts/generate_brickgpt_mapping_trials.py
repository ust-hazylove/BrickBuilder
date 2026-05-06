import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple


LINE_RE = re.compile(
    r"^\s*(\d+)x(\d+)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$"
)

PART_MAP = {
    "1x1": "3005.dat",
    "1x2": "3004.dat",
    "1x3": "3622.dat",
    "1x4": "3010.dat",
    "1x6": "3009.dat",
    "1x8": "3008.dat",
    "2x2": "3003.dat",
    "2x3": "3002.dat",
    "2x4": "3001.dat",
    "2x6": "2456.dat",
    "2x8": "3007.dat",
}

STUD_SPACING = 20.0
BRICK_HEIGHT = 24.0
DEFAULT_COLOR = 4


def roty(deg: int) -> Tuple[int, ...]:
    d = deg % 360
    if d == 0:
        return (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if d == 90:
        return (0, 0, 1, 0, 1, 0, -1, 0, 0)
    raise ValueError("rotation must be 0 or 90")


def canonical_part(h: int, w: int) -> Tuple[str, int]:
    if h <= w:
        return f"{h}x{w}", 0
    return f"{w}x{h}", 1


def parse_text(text: str) -> List[Tuple[int, int, int, int, int]]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue
        rows.append(tuple(int(v) for v in match.groups()))
    if not rows:
        raise ValueError("No valid BrickGPT lines found.")
    return rows


def write_mpd(path: Path, bricks: List[Dict]) -> None:
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


def make_variant_builder(
    *,
    planar_axes: str,
    footprint_mode: str,
    y_sign: int,
) -> Callable[[List[Tuple[int, int, int, int, int]]], List[Dict]]:
    def build(rows: List[Tuple[int, int, int, int, int]]) -> List[Dict]:
        bricks: List[Dict] = []
        for h_raw, w_raw, x, y, z in rows:
            part_name, ori_quarter = canonical_part(h_raw, w_raw)
            file_name = PART_MAP[part_name]

            if footprint_mode == "hw":
                sx, sy = h_raw, w_raw
            elif footprint_mode == "wh":
                sx, sy = w_raw, h_raw
            else:
                raise ValueError(footprint_mode)

            if planar_axes == "x_to_x_y_to_z":
                x_ldr = (x + sx / 2.0) * STUD_SPACING
                z_ldr = (y + sy / 2.0) * STUD_SPACING
            elif planar_axes == "x_to_z_y_to_x":
                x_ldr = (y + sy / 2.0) * STUD_SPACING
                z_ldr = (x + sx / 2.0) * STUD_SPACING
            else:
                raise ValueError(planar_axes)

            y_ldr = y_sign * z * BRICK_HEIGHT
            bricks.append(
                {
                    "id": len(bricks),
                    "file": file_name,
                    "color": DEFAULT_COLOR,
                    "pos": (x_ldr, y_ldr, z_ldr),
                    "rot": list(roty(ori_quarter * 90)),
                    "grid": (h_raw, w_raw, x, y, z),
                }
            )
        return bricks

    return build


VARIANTS = {
    "v1_axes_xz_foot_hw_ypos": make_variant_builder(planar_axes="x_to_x_y_to_z", footprint_mode="hw", y_sign=1),
    "v2_axes_xz_foot_wh_ypos": make_variant_builder(planar_axes="x_to_x_y_to_z", footprint_mode="wh", y_sign=1),
    "v3_axes_zx_foot_hw_ypos": make_variant_builder(planar_axes="x_to_z_y_to_x", footprint_mode="hw", y_sign=1),
    "v4_axes_zx_foot_wh_ypos": make_variant_builder(planar_axes="x_to_z_y_to_x", footprint_mode="wh", y_sign=1),
    "v5_axes_xz_foot_hw_yneg": make_variant_builder(planar_axes="x_to_x_y_to_z", footprint_mode="hw", y_sign=-1),
    "v6_axes_xz_foot_wh_yneg": make_variant_builder(planar_axes="x_to_x_y_to_z", footprint_mode="wh", y_sign=-1),
    "v7_axes_zx_foot_hw_yneg": make_variant_builder(planar_axes="x_to_z_y_to_x", footprint_mode="hw", y_sign=-1),
    "v8_axes_zx_foot_wh_yneg": make_variant_builder(planar_axes="x_to_z_y_to_x", footprint_mode="wh", y_sign=-1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BrickGPT coordinate mapping trial MPDs.")
    parser.add_argument("--input_txt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="brickgpt_trial")
    args = parser.parse_args()

    input_path = Path(args.input_txt).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = parse_text(input_path.read_text(encoding="utf-8"))

    manifest = {}
    for variant_name, builder in VARIANTS.items():
        bricks = builder(rows)
        out_path = output_dir / f"{args.prefix}__{variant_name}.mpd"
        write_mpd(out_path, bricks)
        manifest[variant_name] = {
            "mpd_path": str(out_path),
            "brick_count": len(bricks),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
