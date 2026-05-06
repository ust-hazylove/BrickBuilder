import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RED_COLOR_ID = "4"
LDU_PER_STUD = 20.0
LDU_PER_BRICK = 24.0


PART_SPECS: Dict[str, Tuple[float, float, float]] = {
    "3005.dat": (1, 1, 1.0),
    "3004.dat": (1, 2, 1.0),
    "3622.dat": (1, 3, 1.0),
    "3010.dat": (1, 4, 1.0),
    "3009.dat": (1, 6, 1.0),
    "3008.dat": (1, 8, 1.0),
    "3003.dat": (2, 2, 1.0),
    "3002.dat": (2, 3, 1.0),
    "3001.dat": (2, 4, 1.0),
    "2456.dat": (2, 6, 1.0),
    "3007.dat": (2, 8, 1.0),
    "3024.dat": (1, 1, 1.0 / 3.0),
    "3023.dat": (1, 2, 1.0 / 3.0),
    "3623.dat": (1, 3, 1.0 / 3.0),
    "3710.dat": (1, 4, 1.0 / 3.0),
    "3666.dat": (1, 6, 1.0 / 3.0),
    "3460.dat": (1, 8, 1.0 / 3.0),
    "3022.dat": (2, 2, 1.0 / 3.0),
    "3021.dat": (2, 3, 1.0 / 3.0),
    "3020.dat": (2, 4, 1.0 / 3.0),
    "3795.dat": (2, 6, 1.0 / 3.0),
    "3034.dat": (2, 8, 1.0 / 3.0),
    "3070b.dat": (1, 1, 1.0 / 3.0),
    "3069b.dat": (1, 2, 1.0 / 3.0),
    "2431.dat": (1, 4, 1.0 / 3.0),
    "3068b.dat": (2, 2, 1.0 / 3.0),
    "2412b.dat": (1, 2, 1.0 / 3.0),
    "3040b.dat": (1, 2, 1.0),
    "3039.dat": (2, 2, 1.0),
    "3678.dat": (2, 2, 1.0),
    "3062b.dat": (1, 1, 1.0),
}


@dataclass
class BrickRecord:
    index: int
    line_index: int
    file_section: str
    color: str
    part: str
    x: float
    y: float
    z: float
    bottom_y: float
    top_y: float
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    supported_by: Optional[int] = None
    connected_above: Optional[int] = None
    floating: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect floating bricks in baseline MPD files, mark them red, and optionally render diagnostics."
    )
    parser.add_argument(
        "--input_root",
        default=r"qualitative_pack_100cases_20260417",
        help="Root directory containing qualitative comparison MPD files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing diagnostic MPDs/reports.")
    parser.add_argument("--render", action="store_true", help="Render generated *_floating_red.mpd files with Blender.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging.")
    parser.add_argument("--resolution", type=int, default=1800, help="Render resolution if --render is set.")
    parser.add_argument(
        "--blender",
        default=r"blender",
        help="Blender executable used when --render is set.",
    )
    parser.add_argument(
        "--render_script",
        default=r"scripts\render_floating_highlights_blender.py",
        help="Blender-side rendering script used when --render is set.",
    )
    return parser.parse_args()


def is_generated(path: Path) -> bool:
    stem = path.stem.lower()
    generated_suffixes = (
        "_floating_red",
        "_floating_highlight",
        "_diagnostic",
        "_preview",
        "_trial",
    )
    return stem.endswith(generated_suffixes)


def should_skip_original(path: Path) -> bool:
    stem = path.stem.lower()
    if is_generated(path):
        return True

    parts = {part.lower() for part in path.parts}
    if "ablation" in parts:
        return stem == "ours_full"

    return stem.startswith("ours")


def iter_target_mpds(root: Path) -> Iterable[Path]:
    skip_dirnames = {
        "_baseline_backup_before_rebrick",
        "_trellis_backup_before_fix",
        "_trellis_backup_before_rebrick",
        "_trellis_backup_before_mesh_rebrick",
        "rebuild_trials",
        "voxel_rebrick_trials",
        "trellis_rebrick_trials",
        "trellis_mesh_rebrick_trials",
        "trellis_mesh_identity_trial",
    }
    for path in sorted(root.rglob("*.mpd")):
        if any(part in skip_dirnames for part in path.parts):
            continue
        if should_skip_original(path):
            continue
        yield path


def parse_part_line(line: str):
    tokens = line.strip().split()
    if len(tokens) < 15 or tokens[0] != "1":
        return None
    part = tokens[14].lower()
    if part not in PART_SPECS:
        return None
    try:
        color = tokens[1]
        x, y, z = [float(v) for v in tokens[2:5]]
        matrix = [float(v) for v in tokens[5:14]]
    except ValueError:
        return None
    return color, x, y, z, matrix, part


def projected_footprint(x: float, y: float, z: float, matrix: List[float], part: str):
    rows, cols, height_bricks = PART_SPECS[part]
    half_x = cols * LDU_PER_STUD / 2.0
    half_z = rows * LDU_PER_STUD / 2.0
    half_y = height_bricks * LDU_PER_BRICK / 2.0

    a, b, c, d, e, f, g, h, i = matrix
    xs, ys, zs = [], [], []
    for lx in (-half_x, half_x):
        for ly in (-half_y, half_y):
            for lz in (-half_z, half_z):
                wx = x + a * lx + b * ly + c * lz
                wy = y + d * lx + e * ly + f * lz
                wz = z + g * lx + h * ly + i * lz
                xs.append(wx)
                ys.append(wy)
                zs.append(wz)
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def parse_mpd(path: Path) -> Tuple[List[str], List[BrickRecord]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    records: List[BrickRecord] = []
    section = "MAIN.ldr"
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("0 FILE "):
            section = stripped[7:].strip() or section
            continue
        parsed = parse_part_line(line)
        if parsed is None:
            continue
        color, x, y, z, matrix, part = parsed
        min_x, max_x, bottom_y, top_y, min_z, max_z = projected_footprint(x, y, z, matrix, part)
        records.append(
            BrickRecord(
                index=len(records),
                line_index=line_index,
                file_section=section,
                color=color,
                part=part,
                x=x,
                y=y,
                z=z,
                bottom_y=bottom_y,
                top_y=top_y,
                min_x=min_x,
                max_x=max_x,
                min_z=min_z,
                max_z=max_z,
            )
        )
    return lines, records


def overlap_amount(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def has_footprint_overlap(upper: BrickRecord, lower: BrickRecord, threshold: float) -> bool:
    ox = overlap_amount(upper.min_x, upper.max_x, lower.min_x, lower.max_x)
    oz = overlap_amount(upper.min_z, upper.max_z, lower.min_z, lower.max_z)
    return (ox * oz) >= threshold


def mark_floating(records: List[BrickRecord], y_tol: float = 2.0, overlap_threshold: float = 1.0) -> List[BrickRecord]:
    if not records:
        return records
    ground_y = min(record.bottom_y for record in records)
    for record in records:
        if abs(record.bottom_y - ground_y) <= y_tol:
            record.floating = False
            continue
        lower_contacts = [
            candidate
            for candidate in records
            if candidate.index != record.index
            and abs(candidate.top_y - record.bottom_y) <= y_tol
            and has_footprint_overlap(record, candidate, overlap_threshold)
        ]
        upper_contacts = [
            candidate
            for candidate in records
            if candidate.index != record.index
            and abs(record.top_y - candidate.bottom_y) <= y_tol
            and has_footprint_overlap(record, candidate, overlap_threshold)
        ]
        if lower_contacts:
            best = max(
                lower_contacts,
                key=lambda candidate: overlap_amount(record.min_x, record.max_x, candidate.min_x, candidate.max_x)
                * overlap_amount(record.min_z, record.max_z, candidate.min_z, candidate.max_z),
            )
            record.supported_by = best.index
        if upper_contacts:
            best = max(
                upper_contacts,
                key=lambda candidate: overlap_amount(record.min_x, record.max_x, candidate.min_x, candidate.max_x)
                * overlap_amount(record.min_z, record.max_z, candidate.min_z, candidate.max_z),
            )
            record.connected_above = best.index

        if record.supported_by is not None or record.connected_above is not None:
            record.floating = False
        else:
            record.floating = True
    return records


def recolor_line_red(line: str) -> str:
    parts = line.split()
    if len(parts) >= 2 and parts[0] == "1":
        parts[1] = RED_COLOR_ID
        return " ".join(parts)
    return line


def write_diagnostics(path: Path, lines: List[str], records: List[BrickRecord], overwrite: bool):
    out_mpd = path.with_name(f"{path.stem}_floating_red.mpd")
    out_json = path.with_name(f"{path.stem}_floating_report.json")
    if out_mpd.exists() and out_json.exists() and not overwrite:
        return out_mpd, out_json

    floating_line_indices = {record.line_index for record in records if record.floating}
    output_lines = [
        recolor_line_red(line) if line_index in floating_line_indices else line
        for line_index, line in enumerate(lines)
    ]
    out_mpd.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    report = {
        "source_mpd": str(path),
        "highlight_mpd": str(out_mpd),
        "total_supported_parts": len(records),
        "floating_count": sum(1 for record in records if record.floating),
        "floating_ratio": (sum(1 for record in records if record.floating) / len(records)) if records else 0.0,
        "floating_bricks": [asdict(record) for record in records if record.floating],
    }
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_mpd, out_json


def write_summary(root: Path, rows: List[dict]):
    csv_path = root / "floating_brick_diagnostics_summary.csv"
    json_path = root / "floating_brick_diagnostics_summary.json"
    fieldnames = [
        "relative_path",
        "method",
        "total_supported_parts",
        "floating_count",
        "floating_ratio",
        "highlight_mpd",
        "report_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def render_outputs(args):
    blender = Path(args.blender)
    render_script = Path(args.render_script)
    if not blender.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender}")
    if not render_script.exists():
        raise FileNotFoundError(f"Render script not found: {render_script}")

    cmd = [
        str(blender),
        "-b",
        "-P",
        str(render_script),
        "--",
        "--input_root",
        str(Path(args.input_root).resolve()),
        "--resolution",
        str(args.resolution),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    root = Path(args.input_root).resolve()
    targets = list(iter_target_mpds(root))
    if args.limit > 0:
        targets = targets[: args.limit]

    rows = []
    for mpd_path in targets:
        lines, records = parse_mpd(mpd_path)
        mark_floating(records)
        out_mpd, out_json = write_diagnostics(mpd_path, lines, records, overwrite=args.overwrite)
        floating_count = sum(1 for record in records if record.floating)
        row = {
            "relative_path": str(mpd_path.relative_to(root)),
            "method": mpd_path.stem,
            "total_supported_parts": len(records),
            "floating_count": floating_count,
            "floating_ratio": round(floating_count / len(records), 6) if records else 0.0,
            "highlight_mpd": str(out_mpd),
            "report_json": str(out_json),
        }
        rows.append(row)
        print(
            f"[diagnose] {row['relative_path']}: "
            f"{row['floating_count']}/{row['total_supported_parts']} floating"
        )

    csv_path, json_path = write_summary(root, rows)
    print(f"[summary] {csv_path}")
    print(f"[summary] {json_path}")

    if args.render:
        render_outputs(args)


if __name__ == "__main__":
    main()
