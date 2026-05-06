import argparse
import json
from pathlib import Path


STUD = 20.0
BRICK_H = 24.0


ROT_Y = {
    0: (1, 0, 0, 0, 1, 0, 0, 0, 1),
    1: (0, 0, 1, 0, 1, 0, -1, 0, 0),
    2: (-1, 0, 0, 0, 1, 0, 0, 0, -1),
    3: (0, 0, -1, 0, 1, 0, 1, 0, 0),
}


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild MPD from bricks.json with a chosen vertical axis.")
    parser.add_argument("--input", required=True, help="Path to *_bricks.json.")
    parser.add_argument("--output_dir", required=True, help="Directory for rebuilt MPD files.")
    parser.add_argument(
        "--axes",
        default="x,y,z",
        help="Comma separated candidate up axes. Default: x,y,z",
    )
    parser.add_argument(
        "--yaw_steps",
        default="0,1,2,3",
        help="Comma separated quarter turns around the vertical axis. Default: 0,1,2,3",
    )
    return parser.parse_args()


def load_bricks(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def choose_plane_axes(up_axis: str):
    up_idx = AXIS_INDEX[up_axis]
    plane = [idx for idx in (0, 1, 2) if idx != up_idx]
    return up_idx, plane[0], plane[1]


def calc_centers(bricks, plane_a, plane_b, up_idx):
    min_a = min(brick["grid_pos"][plane_a] for brick in bricks)
    min_b = min(brick["grid_pos"][plane_b] for brick in bricks)
    max_a = max(brick["grid_pos"][plane_a] + brick["size"][0] for brick in bricks)
    max_b = max(brick["grid_pos"][plane_b] + brick["size"][1] for brick in bricks)
    max_up = max(brick["grid_pos"][up_idx] for brick in bricks)
    center_a = (min_a + max_a) * 0.5
    center_b = (min_b + max_b) * 0.5
    return center_a, center_b, max_up


def format_line(color, x, y, z, rot, part_file):
    vals = [color, x, y, z, *rot, part_file]
    return "1 " + " ".join(str(v) for v in vals)


def rebuild_lines(bricks, up_axis: str, yaw_step: int):
    up_idx, plane_a, plane_b = choose_plane_axes(up_axis)
    center_a, center_b, max_up = calc_centers(bricks, plane_a, plane_b, up_idx)

    lines = [
        "0 FILE MAIN.ldr",
        "0 Name: MAIN.ldr",
        "0 Author: Codex",
        f"0 REBUILT_FROM_BRICKS_JSON up_axis={up_axis} yaw_step={yaw_step}",
    ]

    for brick in bricks:
        gx, gy, gz = brick["grid_pos"]
        size_x, size_y = brick["size"]
        grid = (gx, gy, gz)

        a0 = grid[plane_a]
        b0 = grid[plane_b]
        h0 = grid[up_idx]

        world_x = ((a0 + size_x * 0.5) - center_a) * STUD
        world_z = ((b0 + size_y * 0.5) - center_b) * STUD
        world_y = ((max_up - h0) + 0.5) * BRICK_H

        base_quarter = int(brick.get("ori_quarter", 0)) % 4
        rot = ROT_Y[(base_quarter + yaw_step) % 4]
        color = int(brick.get("color", 15))
        part_file = brick["file"]
        lines.append(format_line(color, round(world_x, 4), round(world_y, 4), round(world_z, 4), rot, part_file))

    return lines


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bricks = load_bricks(input_path)
    stem = input_path.stem.replace("_bricks", "")
    axes = [axis.strip() for axis in args.axes.split(",") if axis.strip()]
    yaw_steps = [int(v.strip()) for v in args.yaw_steps.split(",") if v.strip()]

    for up_axis in axes:
        for yaw_step in yaw_steps:
            out_path = output_dir / f"{stem}__up_{up_axis}__yaw_{yaw_step * 90}.mpd"
            lines = rebuild_lines(bricks, up_axis, yaw_step)
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"WRITE {out_path}")


if __name__ == "__main__":
    main()
