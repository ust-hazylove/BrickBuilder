import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


NAME_MAP = {
    "3005.dat": "Brick 1 x 1",
    "3004.dat": "Brick 1 x 2",
    "3622.dat": "Brick 1 x 3",
    "3010.dat": "Brick 1 x 4",
    "3009.dat": "Brick 1 x 6",
    "3008.dat": "Brick 1 x 8",
    "3003.dat": "Brick 2 x 2",
    "3002.dat": "Brick 2 x 3",
    "3001.dat": "Brick 2 x 4",
    "2456.dat": "Brick 2 x 6",
    "3007.dat": "Brick 2 x 8",
}


STUD_SPACING = 20.0
BRICK_HEIGHT = 24.0
DEFAULT_COLOR = 15


def roty(deg: int) -> Tuple[int, ...]:
    d = deg % 360
    if d == 0:
        return (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if d == 90:
        return (0, 0, 1, 0, 1, 0, -1, 0, 0)
    if d == 180:
        return (-1, 0, 0, 0, 1, 0, 0, 0, -1)
    if d == 270:
        return (0, 0, -1, 0, 1, 0, 1, 0, 0)
    raise ValueError("rotation must be multiple of 90")


def matmul3(m1: Tuple[int, ...], m2: Tuple[int, ...]) -> Tuple[int, ...]:
    a1, b1, c1, d1, e1, f1, g1, h1, i1 = m1
    a2, b2, c2, d2, e2, f2, g2, h2, i2 = m2
    return (
        a1 * a2 + b1 * d2 + c1 * g2,
        a1 * b2 + b1 * e2 + c1 * h2,
        a1 * c2 + b1 * f2 + c1 * i2,
        d1 * a2 + e1 * d2 + f1 * g2,
        d1 * b2 + e1 * e2 + f1 * h2,
        d1 * c2 + e1 * f2 + f1 * i2,
        g1 * a2 + h1 * d2 + i1 * g2,
        g1 * b2 + h1 * e2 + i1 * h2,
        g1 * c2 + h1 * f2 + i1 * i2,
    )


def canonical_size(h: int, w: int) -> Tuple[str, int]:
    type_name = f"{min(h, w)}x{max(h, w)}"
    ori_quarter = 1 if h < w else 0
    return type_name, ori_quarter


def parse_brickgpt_lines(text: str, mirror_z: bool = False) -> List[Dict]:
    bricks: List[Dict] = []
    parsed: List[Tuple[int, int, int, int, int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            continue

        parsed.append(tuple(int(v) for v in match.groups()))

    if not parsed:
        return bricks

    max_z = max(z for _, _, _, _, z in parsed)
    for h, w, x, y, z in parsed:
        if mirror_z:
            z = max_z - z
        type_name, ori_quarter = canonical_size(h, w)
        if type_name not in PART_MAP:
            raise ValueError(f"Unsupported brick size from BrickGPT text: {h}x{w}")

        # BrickGPT uses (x, y, z) grid coordinates where z is vertical.
        # Match BrickGPT's direct grid convention:
        # - h spans grid x, w spans grid y.
        # - LDraw Y stores the vertical layer center.
        # - A standard LDraw brick with identity rotation already has studs upward.
        #   Since LDraw's vertical axis is Y, horizontal reorientation must rotate
        #   around Y; rotating around Z would tip bricks onto their sides.
        x_ldr = (x + h / 2.0) * STUD_SPACING
        z_ldr = (y + w / 2.0) * STUD_SPACING
        y_ldr = (z + 0.5) * BRICK_HEIGHT
        file_name = PART_MAP[type_name]
        rotation = roty(ori_quarter * 90)

        bricks.append(
            {
                "id": len(bricks),
                "file": file_name,
                "name": NAME_MAP[file_name],
                "color": DEFAULT_COLOR,
                "pos": (x_ldr, y_ldr, z_ldr),
                "rot": list(rotation),
                "type_name": type_name,
                "struct_type": type_name,
                "size": (h, w),
                "grid_pos": (x, y, z),
                "ori_quarter": ori_quarter,
                "semantic_label": 0,
            }
        )
    return bricks


def load_input_text(input_txt: Optional[Path], input_inline: Optional[str]) -> str:
    if input_inline:
        return input_inline
    if input_txt is None:
        raise ValueError("Either --input_txt or --input_inline must be provided.")
    return input_txt.read_text(encoding="utf-8-sig")


def write_ldr(bricks: List[Dict], path: Path) -> None:
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


def build_mpd(bricks: List[Dict], path: Path) -> None:
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


def convert_text(
    text: str,
    output_dir: Path,
    output_stem: str = "brickgpt",
    input_path: Optional[Path] = None,
    mirror_z: bool = False,
) -> Dict:
    bricks = parse_brickgpt_lines(text, mirror_z=mirror_z)
    if not bricks:
        raise ValueError("No valid BrickGPT brick lines were parsed from the input.")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{output_stem}_bricks.json"
    ldr_path = output_dir / f"{output_stem}.ldr"
    mpd_path = output_dir / f"{output_stem}.mpd"
    summary_path = output_dir / "summary.json"

    json_path.write_text(json.dumps(bricks, indent=2, ensure_ascii=False), encoding="utf-8")
    write_ldr(bricks, ldr_path)
    build_mpd(bricks, mpd_path)

    summary = {
        "input_text_path": str(input_path) if input_path else None,
        "brick_count": len(bricks),
        "json_path": str(json_path),
        "ldr_path": str(ldr_path),
        "mpd_path": str(mpd_path),
        "output_stem": output_stem,
        "mirror_z": mirror_z,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def convert_file(
    input_path: Path,
    output_dir: Path,
    output_stem: str = "brickgpt",
    mirror_z: bool = False,
) -> Dict:
    text = input_path.read_text(encoding="utf-8-sig")
    return convert_text(
        text,
        output_dir=output_dir,
        output_stem=output_stem,
        input_path=input_path,
        mirror_z=mirror_z,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BrickGPT text brick output to unified JSON/LDR/MPD.")
    parser.add_argument("--input_txt", default=None, help="Path to the BrickGPT brick text file.")
    parser.add_argument("--input_inline", default=None, help="Inline BrickGPT brick text content.")
    parser.add_argument("--output_dir", required=True, help="Directory for converted outputs.")
    parser.add_argument(
        "--output_stem",
        default="brickgpt",
        help="Unified basename for output files, e.g. brickgpt / brickgpt_text / case001_brickgpt.",
    )
    parser.add_argument(
        "--mirror_z",
        action="store_true",
        help="Mirror BrickGPT vertical layers before mapping z to LDraw Y. Use when BrickGPT outputs top-down layers.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_txt).resolve() if args.input_txt else None
    text = load_input_text(input_path, args.input_inline)
    summary = convert_text(
        text=text,
        output_dir=Path(args.output_dir).resolve(),
        output_stem=args.output_stem,
        input_path=input_path,
        mirror_z=args.mirror_z,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
