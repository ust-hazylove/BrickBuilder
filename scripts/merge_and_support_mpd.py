import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


LDU_PER_STUD = 20.0
LDU_PER_BRICK = 24.0


PART_TO_SIZE = {
    "3005.dat": (1, 1),
    "3004.dat": (2, 1),
    "3622.dat": (3, 1),
    "3010.dat": (4, 1),
    "3009.dat": (6, 1),
    "3008.dat": (8, 1),
    "3003.dat": (2, 2),
    "3002.dat": (3, 2),
    "3001.dat": (4, 2),
    "2456.dat": (6, 2),
    "3007.dat": (8, 2),
}

SIZE_TO_PART_ROT = {}
for part, (sx, sz) in PART_TO_SIZE.items():
    SIZE_TO_PART_ROT.setdefault((sx, sz), (part, 0))
    SIZE_TO_PART_ROT.setdefault((sz, sx), (part, 1))
ALLOWED_SIZES = sorted(SIZE_TO_PART_ROT, key=lambda size: (size[0] * size[1], max(size), min(size)), reverse=True)


def yaw_y_matrix(quarter_turn):
    if quarter_turn % 4 == 0:
        return (1, 0, 0, 0, 1, 0, 0, 0, 1)
    if quarter_turn % 4 == 1:
        return (0, 0, 1, 0, 1, 0, -1, 0, 0)
    if quarter_turn % 4 == 2:
        return (-1, 0, 0, 0, 1, 0, 0, 0, -1)
    return (0, 0, -1, 0, 1, 0, 1, 0, 0)


def make_brick(x, z, layer, sx, sz):
    part, rot_quarter = SIZE_TO_PART_ROT[(sx, sz)]
    return {"x": x, "z": z, "layer": layer, "sx": sx, "sz": sz, "part": part, "rot_quarter": rot_quarter}


def parse_mpd_to_voxels(path: Path):
    occupied = set()
    color_counts = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        tokens = line.split()
        if len(tokens) < 15 or tokens[0] != "1":
            continue
        part = tokens[14].lower()
        if part not in PART_TO_SIZE:
            continue
        matrix = tuple(float(v) for v in tokens[5:14])
        if matrix != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            raise ValueError(f"Only identity-rotation MPDs are supported by this utility. Found {matrix}")
        color = tokens[1]
        x, y, z = (float(tokens[2]), float(tokens[3]), float(tokens[4]))
        sx, sz = PART_TO_SIZE[part]
        min_x = round((x - sx * LDU_PER_STUD / 2.0) / LDU_PER_STUD)
        min_z = round((z - sz * LDU_PER_STUD / 2.0) / LDU_PER_STUD)
        layer = round((y - LDU_PER_BRICK / 2.0) / LDU_PER_BRICK)
        color_counts[color] = color_counts.get(color, 0) + 1
        for gx in range(min_x, min_x + sx):
            for gz in range(min_z, min_z + sz):
                occupied.add((gx, gz, layer))
    if not occupied:
        raise ValueError(f"No supported standard bricks found in {path}")
    color = max(color_counts.items(), key=lambda item: item[1])[0]
    return occupied, color


def normalize_occupied(occupied):
    min_x = min(x for x, _, _ in occupied)
    min_z = min(z for _, z, _ in occupied)
    min_layer = min(layer for _, _, layer in occupied)
    normalized = {(x - min_x, z - min_z, layer - min_layer) for x, z, layer in occupied}
    return normalized, (min_x, min_z, min_layer)


def denormalize_bricks(bricks, offset):
    ox, oz, ol = offset
    output = []
    for brick in bricks:
        item = dict(brick)
        item["x"] += ox
        item["z"] += oz
        item["layer"] += ol
        output.append(item)
    return output


def fill_vertical_supports(occupied):
    repaired = set(occupied)
    for x, z, layer in list(occupied):
        for support_layer in range(layer):
            repaired.add((x, z, support_layer))
    return repaired


def cell_has_vertical_contact(occupied, cell):
    x, z, layer = cell
    return layer == 0 or (x, z, layer - 1) in occupied or (x, z, layer + 1) in occupied


def keep_largest_component(occupied):
    remaining = set(occupied)
    components = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        queue = deque([seed])
        while queue:
            x, z, y = queue.popleft()
            for nb in ((x + 1, z, y), (x - 1, z, y), (x, z + 1, y), (x, z - 1, y), (x, z, y + 1), (x, z, y - 1)):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return max(components, key=len), sorted((len(c) for c in components), reverse=True)


def can_place(layer_grid, used, x, z, sx, sz):
    h, w = layer_grid.shape
    if x + sx > w or z + sz > h:
        return False
    return bool(layer_grid[z:z + sz, x:x + sx].all()) and not bool(used[z:z + sz, x:x + sx].any())


def greedy_merge_layer(layer_grid, layer):
    used = np.zeros_like(layer_grid, dtype=bool)
    bricks = []
    h, w = layer_grid.shape
    for z in range(h):
        for x in range(w):
            if not layer_grid[z, x] or used[z, x]:
                continue
            placed = False
            for sx, sz in ALLOWED_SIZES:
                if can_place(layer_grid, used, x, z, sx, sz):
                    used[z:z + sz, x:x + sx] = True
                    bricks.append(make_brick(x, z, layer, sx, sz))
                    placed = True
                    break
            if not placed:
                used[z, x] = True
                bricks.append(make_brick(x, z, layer, 1, 1))
    return bricks


def rect_cells(x, z, sx, sz, layer):
    return {(gx, gz, layer) for gx in range(x, x + sx) for gz in range(z, z + sz)}


def can_place_cells(occupied, used_cells, x, z, sx, sz, layer):
    cells = rect_cells(x, z, sx, sz, layer)
    return cells.issubset(occupied) and not (cells & used_cells)


def support_aware_merge_layer(occupied, layer):
    layer_cells = {(x, z, y) for x, z, y in occupied if y == layer}
    used = set()
    bricks = []

    def best_contact_area(cell):
        cx, cz, _ = cell
        best = 1
        for sx, sz in ALLOWED_SIZES:
            for x0 in range(cx - sx + 1, cx + 1):
                for z0 in range(cz - sz + 1, cz + 1):
                    cells = rect_cells(x0, z0, sx, sz, layer)
                    if cells.issubset(occupied) and any(cell_has_vertical_contact(occupied, item) for item in cells):
                        best = max(best, sx * sz)
        return best

    # Handle unsupported cells first so they can borrow support through a same-layer larger brick.
    ordered = sorted(
        layer_cells,
        key=lambda cell: (
            cell_has_vertical_contact(occupied, cell),
            -best_contact_area(cell),
            cell[1],
            cell[0],
        ),
    )
    for cell in ordered:
        if cell in used:
            continue
        cx, cz, _ = cell
        candidates = []
        for sx, sz in ALLOWED_SIZES:
            for x0 in range(cx - sx + 1, cx + 1):
                for z0 in range(cz - sz + 1, cz + 1):
                    if can_place_cells(occupied, used, x0, z0, sx, sz, layer):
                        cells = rect_cells(x0, z0, sx, sz, layer)
                        has_contact = any(cell_has_vertical_contact(occupied, item) for item in cells)
                        candidates.append((has_contact, sx * sz, max(sx, sz), min(sx, sz), x0, z0, sx, sz))
        if candidates:
            _, _, _, _, x, z, sx, sz = max(candidates)
        else:
            x, z, sx, sz = cx, cz, 1, 1
        used.update(rect_cells(x, z, sx, sz, layer))
        bricks.append(make_brick(x, z, layer, sx, sz))
    return bricks


def merge_occupied(occupied):
    max_x = max(x for x, _, _ in occupied)
    max_z = max(z for _, z, _ in occupied)
    max_layer = max(layer for _, _, layer in occupied)
    bricks = []
    for layer in range(max_layer + 1):
        grid = np.zeros((max_z + 1, max_x + 1), dtype=bool)
        for x, z, y in occupied:
            if y == layer:
                grid[z, x] = True
        bricks.extend(greedy_merge_layer(grid, layer))
    return bricks


def merge_occupied_preserving_support(occupied):
    max_layer = max(layer for _, _, layer in occupied)
    bricks = []
    for layer in range(max_layer + 1):
        bricks.extend(exact_cover_merge_layer(occupied, layer))
    return bricks


def exact_cover_merge_layer(occupied, layer):
    layer_cells = sorted((x, z, y) for x, z, y in occupied if y == layer)
    if not layer_cells:
        return []
    cell_to_row = {cell: idx for idx, cell in enumerate(layer_cells)}
    candidates = []
    for cell in layer_cells:
        cx, cz, _ = cell
        for sx, sz in ALLOWED_SIZES:
            for x0 in range(cx - sx + 1, cx + 1):
                for z0 in range(cz - sz + 1, cz + 1):
                    cells = rect_cells(x0, z0, sx, sz, layer)
                    if not cells.issubset(occupied):
                        continue
                    if not any(cell_has_vertical_contact(occupied, item) for item in cells):
                        continue
                    # Avoid duplicate rectangles discovered from multiple cells.
                    key = (x0, z0, sx, sz)
                    if key not in {candidate[0] for candidate in candidates}:
                        candidates.append((key, cells))

    if not candidates:
        return support_aware_merge_layer(occupied, layer)

    matrix = lil_matrix((len(layer_cells), len(candidates)), dtype=float)
    for col, (_, cells) in enumerate(candidates):
        for cell in cells:
            matrix[cell_to_row[cell], col] = 1.0
    constraints = LinearConstraint(matrix.tocsr(), np.ones(len(layer_cells)), np.ones(len(layer_cells)))
    costs = np.array([1.0 - 0.001 * len(cells) for _, cells in candidates])
    result = milp(
        c=costs,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraints,
        options={"time_limit": 60},
    )
    if not result.success:
        return repair_floating_bricks_preserve_occupancy(support_aware_merge_layer(occupied, layer), occupied)

    bricks = []
    for selected, (key, _) in zip(result.x, candidates):
        if selected > 0.5:
            x, z, sx, sz = key
            bricks.append(make_brick(x, z, layer, sx, sz))
    return bricks


def brick_cells(brick):
    return rect_cells(brick["x"], brick["z"], brick["sx"], brick["sz"], brick["layer"])


def brick_has_vertical_contact(occupied, brick):
    return brick["layer"] == 0 or any(cell_has_vertical_contact(occupied, cell) for cell in brick_cells(brick))


def repair_floating_bricks_preserve_occupancy(bricks, occupied, max_passes=12):
    bricks = list(bricks)
    for _ in range(max_passes):
        floating_index = next(
            (idx for idx, brick in enumerate(bricks) if not brick_has_vertical_contact(occupied, brick)),
            None,
        )
        if floating_index is None:
            return bricks

        floating = bricks[floating_index]
        layer = floating["layer"]
        candidates = []
        for fx, fz, _ in brick_cells(floating):
            for sx, sz in ALLOWED_SIZES:
                for x0 in range(fx - sx + 1, fx + 1):
                    for z0 in range(fz - sz + 1, fz + 1):
                        candidate_cells = rect_cells(x0, z0, sx, sz, layer)
                        if not candidate_cells.issubset(occupied):
                            continue
                        if not any(cell_has_vertical_contact(occupied, cell) for cell in candidate_cells):
                            continue
                        overlap_indices = [
                            idx
                            for idx, brick in enumerate(bricks)
                            if brick["layer"] == layer and brick_cells(brick) & candidate_cells
                        ]
                        removed_cells = set()
                        for idx in overlap_indices:
                            removed_cells.update(brick_cells(bricks[idx]))
                        leftovers = removed_cells - candidate_cells
                        candidates.append(
                            (
                                sx * sz,
                                -len(leftovers),
                                -len(overlap_indices),
                                x0,
                                z0,
                                sx,
                                sz,
                                overlap_indices,
                                leftovers,
                            )
                        )
        if not candidates:
            return bricks

        _, _, _, x, z, sx, sz, overlap_indices, leftovers = max(candidates)
        overlap_set = set(overlap_indices)
        kept = [brick for idx, brick in enumerate(bricks) if idx not in overlap_set]
        kept.append(make_brick(x, z, layer, sx, sz))
        if leftovers:
            kept.extend(support_aware_merge_layer(leftovers, layer))
        bricks = kept
    return bricks


def write_mpd(bricks, color, output_path: Path):
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("0 FILE MAIN.ldr\n")
        handle.write("0 Name: MAIN.ldr\n")
        handle.write("0 Author: Img2Build merge_and_support_mpd.py\n")
        handle.write("0 MERGED_AND_SUPPORT_REPAIRED 1\n")
        for brick in bricks:
            x = (brick["x"] + brick["sx"] / 2.0) * LDU_PER_STUD
            z = (brick["z"] + brick["sz"] / 2.0) * LDU_PER_STUD
            y = (brick["layer"] + 0.5) * LDU_PER_BRICK
            rot = yaw_y_matrix(brick.get("rot_quarter", 0))
            handle.write(
                f"1 {color} {x:.3f} {y:.3f} {z:.3f} "
                f"{rot[0]} {rot[1]} {rot[2]} {rot[3]} {rot[4]} {rot[5]} {rot[6]} {rot[7]} {rot[8]} {brick['part']}\n"
            )
            handle.write("0 STEP\n")
        handle.write("0 NOFILE\n")


def support_report(bricks):
    cells = set()
    for brick in bricks:
        for x in range(brick["x"], brick["x"] + brick["sx"]):
            for z in range(brick["z"], brick["z"] + brick["sz"]):
                cells.add((x, z, brick["layer"]))
    unsupported = []
    for idx, brick in enumerate(bricks):
        if brick["layer"] == 0:
            continue
        has_support = False
        has_above = False
        for x in range(brick["x"], brick["x"] + brick["sx"]):
            for z in range(brick["z"], brick["z"] + brick["sz"]):
                has_support = has_support or ((x, z, brick["layer"] - 1) in cells)
                has_above = has_above or ((x, z, brick["layer"] + 1) in cells)
        if not has_support and not has_above:
            unsupported.append(idx)
    _, component_sizes = keep_largest_component(cells)
    return {
        "brick_count": len(bricks),
        "occupied_voxels": len(cells),
        "floating_like_brick_count": len(unsupported),
        "component_count": len(component_sizes),
        "component_sizes": component_sizes,
    }


def brick_occupied_cells(bricks):
    cells = set()
    for brick in bricks:
        for x in range(brick["x"], brick["x"] + brick["sx"]):
            for z in range(brick["z"], brick["z"] + brick["sz"]):
                cells.add((x, z, brick["layer"]))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_mpd", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--preserve-occupancy",
        action="store_true",
        help="Do not add vertical support voxels; only repartition existing occupied cells into larger interlocking bricks.",
    )
    args = parser.parse_args()

    input_mpd = args.input_mpd.resolve()
    output = args.output or input_mpd.with_name(f"{input_mpd.stem}_merged_supported.mpd")
    report_path = args.report or input_mpd.with_name(f"{input_mpd.stem}_merged_supported_report.json")

    occupied, color = parse_mpd_to_voxels(input_mpd)
    original_count = sum(1 for line in input_mpd.read_text(encoding="utf-8", errors="ignore").splitlines() if line.startswith("1 "))
    normalized, offset = normalize_occupied(occupied)
    if args.preserve_occupancy:
        target = normalized
        connected, component_sizes_before_keep = keep_largest_component(target)
        if len(connected) != len(target):
            print("[warn] Input occupancy has multiple disconnected voxel components; preserving occupancy keeps them.")
            connected = target
        bricks = denormalize_bricks(merge_occupied_preserving_support(connected), offset)
    else:
        supported = fill_vertical_supports(normalized)
        connected, component_sizes_before_keep = keep_largest_component(supported)
        bricks = denormalize_bricks(merge_occupied(connected), offset)
    write_mpd(bricks, color, output)

    report = support_report(bricks)
    rebuilt_cells_normalized, _ = normalize_occupied(brick_occupied_cells(bricks))
    report.update(
        {
            "input_mpd": str(input_mpd),
            "output_mpd": str(output),
            "original_brick_count": original_count,
            "original_occupied_voxels": len(occupied),
            "output_occupied_voxels": len(brick_occupied_cells(bricks)),
            "preserve_occupancy": bool(args.preserve_occupancy),
            "occupancy_exact_match": rebuilt_cells_normalized == normalized,
            "component_sizes_before_keep": component_sizes_before_keep,
            "dominant_color": color,
        }
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
