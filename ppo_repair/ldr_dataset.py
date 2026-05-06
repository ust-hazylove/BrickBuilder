import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LDU_PER_STUD = 20.0
LDU_PER_BRICK = 24.0
HALFGRID = 2

PART_SPECS: Dict[str, Tuple[int, int, float]] = {
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
    "3040b.dat": (1, 2, 1.0),
    "3039.dat": (2, 2, 1.0),
}


@dataclass
class ParsedBrick:
    part: str
    x: int
    y: int
    z: int
    dx: int
    dy: int
    dz: int


def nearest_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def classify_upright_y(matrix: np.ndarray, tolerance_deg: float = 8.0) -> Optional[int]:
    snapped = nearest_rotation_matrix(matrix)
    y_col = snapped[:, 1]
    cos_plus = float(np.clip(y_col @ np.array([0.0, 1.0, 0.0], dtype=float), -1.0, 1.0))
    angle_plus = math.degrees(math.acos(cos_plus))
    if angle_plus > tolerance_deg:
        cos_minus = float(np.clip(y_col @ np.array([0.0, -1.0, 0.0], dtype=float), -1.0, 1.0))
        angle_minus = math.degrees(math.acos(cos_minus))
        if angle_minus > tolerance_deg:
            return None
        snapped = snapped @ np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

    theta = math.degrees(math.atan2(float(snapped[0, 2]), float(snapped[0, 0])))
    quarter_turn = int(round(theta / 90.0)) % 4
    if abs(theta - quarter_turn * 90.0) > tolerance_deg:
        return None
    return quarter_turn


def snap_half_stud(value_stud: float) -> int:
    return int(round(value_stud * HALFGRID))


def parse_ldr_file(path: str, verbose: bool = False) -> List[ParsedBrick]:
    parsed: List[ParsedBrick] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("0"):
                continue
            tokens = line.split()
            if tokens[0] != "1" or len(tokens) < 15:
                continue

            x, y, z = map(float, tokens[2:5])
            a, b, c, d, e, f, g, h, i = map(float, tokens[5:14])
            part = tokens[14].lower()
            spec = PART_SPECS.get(part)
            if spec is None:
                if verbose:
                    print(f"[LDR] Skip unsupported part {part} at {path}:{line_number}")
                continue

            rows, cols, height = spec
            rotation = classify_upright_y(np.array([[a, b, c], [d, e, f], [g, h, i]], dtype=float))
            if rotation is None:
                if verbose:
                    print(f"[LDR] Skip non-upright part {part} at {path}:{line_number}")
                continue

            if rotation in (1, 3):
                rows, cols = cols, rows

            center_x_h = snap_half_stud(x / LDU_PER_STUD)
            center_y_h = snap_half_stud(z / LDU_PER_STUD)
            x0_h = center_x_h - (cols * HALFGRID // 2)
            y0_h = center_y_h - (rows * HALFGRID // 2)
            z0_b = y / LDU_PER_BRICK

            parsed.append(
                ParsedBrick(
                    part=part,
                    x=int(round(x0_h / HALFGRID)),
                    y=int(round(y0_h / HALFGRID)),
                    z=int(round(z0_b)),
                    dx=int(cols),
                    dy=int(rows),
                    dz=max(1, int(round(height))),
                )
            )
    return parsed


def bricks_to_voxel_grid(bricks: Sequence[ParsedBrick]) -> np.ndarray:
    if not bricks:
        return np.zeros((1, 1, 1), dtype=np.uint8)

    min_x = min(brick.x for brick in bricks)
    min_y = min(brick.y for brick in bricks)
    min_z = min(brick.z for brick in bricks)
    shifted = [
        ParsedBrick(
            part=brick.part,
            x=brick.x - min_x,
            y=brick.y - min_y,
            z=brick.z - min_z,
            dx=brick.dx,
            dy=brick.dy,
            dz=brick.dz,
        )
        for brick in bricks
    ]

    max_x = max(brick.x + brick.dx for brick in shifted)
    max_y = max(brick.y + brick.dy for brick in shifted)
    max_z = max(brick.z + brick.dz for brick in shifted)
    grid = np.zeros((max_x, max_y, max_z), dtype=np.uint8)
    for brick in shifted:
        grid[brick.x:brick.x + brick.dx, brick.y:brick.y + brick.dy, brick.z:brick.z + brick.dz] = 1
    return grid


def shrink_binary_grid(voxel_grid: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(target_shape, dtype=np.uint8)
    for ox in range(target_shape[0]):
        x0 = int(math.floor(ox * voxel_grid.shape[0] / target_shape[0]))
        x1 = int(math.ceil((ox + 1) * voxel_grid.shape[0] / target_shape[0]))
        for oy in range(target_shape[1]):
            y0 = int(math.floor(oy * voxel_grid.shape[1] / target_shape[1]))
            y1 = int(math.ceil((oy + 1) * voxel_grid.shape[1] / target_shape[1]))
            for oz in range(target_shape[2]):
                z0 = int(math.floor(oz * voxel_grid.shape[2] / target_shape[2]))
                z1 = int(math.ceil((oz + 1) * voxel_grid.shape[2] / target_shape[2]))
                block = voxel_grid[x0:max(x1, x0 + 1), y0:max(y1, y0 + 1), z0:max(z1, z0 + 1)]
                output[ox, oy, oz] = 1 if np.any(block) else 0
    return output


def fit_voxels_to_grid(voxel_grid: np.ndarray, grid_size: int = 32) -> np.ndarray:
    coords = np.argwhere(voxel_grid > 0)
    if coords.size == 0:
        return np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    cropped = voxel_grid[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    dims = np.array(cropped.shape, dtype=int)
    if np.any(dims > grid_size):
        scale = min(grid_size / max(int(dims[0]), 1), grid_size / max(int(dims[1]), 1), grid_size / max(int(dims[2]), 1))
        scaled_dims = np.maximum(1, np.floor(dims * scale).astype(int))
        cropped = shrink_binary_grid(cropped, tuple(int(v) for v in scaled_dims))
        dims = np.array(cropped.shape, dtype=int)

    fitted = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
    starts = ((grid_size - dims) // 2).astype(int)
    fitted[
        starts[0]:starts[0] + dims[0],
        starts[1]:starts[1] + dims[1],
        starts[2]:starts[2] + dims[2],
    ] = cropped
    return fitted


def load_ldr_as_voxels(path: str, grid_size: int = 32, verbose: bool = False) -> np.ndarray:
    bricks = parse_ldr_file(path, verbose=verbose)
    raw_grid = bricks_to_voxel_grid(bricks)
    return fit_voxels_to_grid(raw_grid, grid_size=grid_size)


def iter_dataset_files(root_dir: str) -> Iterable[str]:
    if not root_dir or not os.path.exists(root_dir):
        return []
    patterns = ("*.npy", "*.npz", "*.ldr")
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(root_dir, pattern)))
    return sorted(files)


def prepare_ldr_dataset(source_root: str, output_root: str, grid_size: int = 32, limit: Optional[int] = None) -> int:
    os.makedirs(output_root, exist_ok=True)
    ldr_files = sorted(glob.glob(os.path.join(source_root, "*.ldr")))
    if limit is not None:
        ldr_files = ldr_files[:limit]

    converted = 0
    for path in ldr_files:
        try:
            voxels = load_ldr_as_voxels(path, grid_size=grid_size)
        except Exception as exc:
            print(f"[LDR] Failed to convert {os.path.basename(path)}: {exc}")
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        np.savez_compressed(os.path.join(output_root, f"{stem}.npz"), voxels=voxels.astype(np.uint8))
        converted += 1
    return converted
