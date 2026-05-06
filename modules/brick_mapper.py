import numpy as np
import pandas as pd
from collections import Counter

import modules.voxel_to_bricks as v2b
from modules.lego_colors import PALETTE


class BrickMapper:
    def __init__(self):
        self.part_map = {
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
            "tile_1x1": "3070b.dat",
            "tile_1x2": "3069b.dat",
            "tile_1x4": "2431.dat",
            "tile_2x2": "3068b.dat",
            "grille_tile_1x2": "2412b.dat",
            "slope_1x2": "3040b.dat",
            "slope_2x2": "3039.dat",
            "corner_slope_2x2": "3678.dat",
            "round_brick_1x1": "3062b.dat",
        }
        self.name_map = {
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
            "3070b.dat": "Tile 1 x 1",
            "3069b.dat": "Tile 1 x 2",
            "2431.dat": "Tile 1 x 4",
            "3068b.dat": "Tile 2 x 2",
            "2412b.dat": "Tile, Modified 1 x 2 Grille",
            "3040b.dat": "Slope Brick 45 2 x 1",
            "3039.dat": "Slope Brick 45 2 x 2",
            "3678.dat": "Slope Brick 45 2 x 2 Corner",
            "3062b.dat": "Brick, Round 1 x 1 Straight Side",
        }
        self.tile_map = {
            (1, 1): ("tile_1x1", 0),
            (2, 1): ("tile_1x2", 0),
            (1, 2): ("tile_1x2", 1),
            (4, 1): ("tile_1x4", 0),
            (1, 4): ("tile_1x4", 1),
            (2, 2): ("tile_2x2", 0),
        }
        self.grille_map = {
            (2, 1): ("grille_tile_1x2", 0),
            (1, 2): ("grille_tile_1x2", 1),
        }
        self.slope_map = {
            (2, 1): "slope_1x2",
            (1, 2): "slope_1x2",
            (2, 2): "slope_2x2",
        }
        self.stud_spacing = 20.0
        self.brick_height = 24.0

    def map_voxels_to_bricks(self, voxel_grid: np.ndarray, color_grid: np.ndarray = None, verbose: bool = True):
        raw_bricks = v2b.merge_all_layers_from_grid(voxel_grid)
        if verbose:
            print(f"  [Mapper] Merged {np.sum(voxel_grid)} voxels into {len(raw_bricks)} bricks.")

        mapped_bricks = []
        for i, brick in enumerate(raw_bricks):
            color_id = self._sample_color(brick, color_grid)
            x_ldr = (brick.x + brick.dx / 2.0) * self.stud_spacing
            z_ldr = (brick.y + brick.dy / 2.0) * self.stud_spacing
            y_ldr = (brick.z + 0.5) * self.brick_height

            ori_quarter = int(brick.rot // 90) % 4 if brick.rot else 0
            if brick.dx < brick.dy:
                ori_quarter = (ori_quarter + 1) % 4

            file_name = self.part_map.get(brick.type, "3001.dat")
            mapped_bricks.append(
                {
                    "id": i,
                    "file": file_name,
                    "name": self.name_map.get(file_name, brick.type),
                    "color": int(color_id),
                    "pos": (x_ldr, y_ldr, z_ldr),
                    "rot": list(v2b.matmul3(v2b.rotz(ori_quarter * 90), v2b.flip_upside())),
                    "type_name": brick.type,
                    "struct_type": brick.type,
                    "size": (int(brick.dx), int(brick.dy)),
                    "grid_pos": (int(brick.x), int(brick.y), int(brick.z)),
                    "ori_quarter": ori_quarter,
                    "semantic_label": 0,
                }
            )
        return mapped_bricks

    def apply_surface_finishing(self, brick_list, voxel_grid: np.ndarray):
        styled = []
        for brick in brick_list:
            updated = dict(brick)
            gx, gy, gz = brick["grid_pos"]
            dx, dy = brick["size"]
            top_exposed = self._is_top_exposed(voxel_grid, gx, gy, gz, dx, dy)
            exposure = self._side_exposure(voxel_grid, gx, gy, gz, dx, dy)

            style_key = None
            corner_pair = self._dominant_corner(exposure)
            if top_exposed:
                dominant_side = max(exposure, key=exposure.get)
                if self._should_use_corner_slope(dx, dy, exposure):
                    style_key = "corner_slope_2x2"
                    updated["ori_quarter"] = self._quarter_turn_for_corner(corner_pair)
                    updated["semantic_label"] = self._semantic_label_for_corner(corner_pair)
                elif self._should_use_slope(dx, dy, exposure):
                    style_key = self.slope_map.get((dx, dy))
                    updated["ori_quarter"] = self._quarter_turn_for_side(dx, dy, dominant_side)
                    updated["semantic_label"] = self._semantic_label_for_side(dominant_side)
                elif self._should_use_grille_tile(dx, dy, exposure):
                    grille_spec = self.grille_map.get((dx, dy))
                    if grille_spec is not None:
                        style_key, extra_turn = grille_spec
                        updated["ori_quarter"] = (brick["ori_quarter"] + extra_turn) % 4
                elif self._should_use_tile(dx, dy, exposure):
                    tile_spec = self.tile_map.get((dx, dy))
                    if tile_spec is not None:
                        style_key, extra_turn = tile_spec
                        updated["ori_quarter"] = (brick["ori_quarter"] + extra_turn) % 4
            elif self._should_use_round_brick(dx, dy, exposure):
                style_key = "round_brick_1x1"

            if style_key is not None:
                file_name = self.part_map.get(style_key, updated["file"])
                updated["file"] = file_name
                updated["name"] = self.name_map.get(file_name, updated["name"])
                updated["type_name"] = style_key

            updated["rot"] = list(v2b.matmul3(v2b.rotz(updated["ori_quarter"] * 90), v2b.flip_upside()))
            styled.append(updated)
        return styled

    def generate_bom(self, brick_list):
        stats = Counter()
        color_map = {c[0]: c[4] for c in PALETTE.colors}

        for brick in brick_list:
            part_name = brick.get("name", "Unknown Brick")
            color_id = brick["color"]
            color_name = color_map.get(color_id, f"Color {color_id}")
            stats[(part_name, color_name, color_id)] += 1

        data = []
        for (part, color, color_id), count in stats.items():
            data.append(
                {
                    "Part Name": part,
                    "Color": color,
                    "Color ID": color_id,
                    "Quantity": count,
                }
            )

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(by=["Part Name", "Quantity"], ascending=[True, False])
        return df

    def _sample_color(self, brick, color_grid):
        color_id = 15
        if color_grid is None:
            return color_id
        try:
            cx = min(int(brick.x + brick.dx // 2), color_grid.shape[0] - 1)
            cy = min(int(brick.y + brick.dy // 2), color_grid.shape[1] - 1)
            cz = min(int(brick.z), color_grid.shape[2] - 1)
            rgb = color_grid[cx, cy, cz]
            if np.mean(rgb) < 250:
                color_id = PALETTE.get_nearest_color_id(rgb)
        except Exception:
            pass
        return color_id

    def _is_top_exposed(self, voxel_grid, gx, gy, gz, dx, dy):
        z_top = gz + 1
        if z_top >= voxel_grid.shape[2]:
            return True
        return not bool(voxel_grid[gx:gx + dx, gy:gy + dy, z_top].any())

    def _side_exposure(self, voxel_grid, gx, gy, gz, dx, dy):
        max_x, max_y, _ = voxel_grid.shape
        exposure = {"left": 0, "right": 0, "front": 0, "back": 0}

        exposure["left"] = dy if gx == 0 else dy - int(voxel_grid[gx - 1, gy:gy + dy, gz].sum())
        exposure["right"] = dy if gx + dx >= max_x else dy - int(voxel_grid[gx + dx, gy:gy + dy, gz].sum())
        exposure["front"] = dx if gy == 0 else dx - int(voxel_grid[gx:gx + dx, gy - 1, gz].sum())
        exposure["back"] = dx if gy + dy >= max_y else dx - int(voxel_grid[gx:gx + dx, gy + dy, gz].sum())
        return exposure

    def _should_use_slope(self, dx, dy, exposure):
        if (dx, dy) not in self.slope_map:
            return False
        return max(exposure.values()) >= max(dx, dy)

    def _should_use_corner_slope(self, dx, dy, exposure):
        if (dx, dy) != (2, 2):
            return False
        ordered = sorted(exposure.items(), key=lambda item: item[1], reverse=True)
        return ordered[0][1] >= 1 and ordered[1][1] >= 1 and self._are_adjacent_sides(ordered[0][0], ordered[1][0])

    def _should_use_grille_tile(self, dx, dy, exposure):
        if (dx, dy) not in self.grille_map:
            return False
        values = sorted(exposure.values(), reverse=True)
        return values[0] <= max(dx, dy) and values[1] <= max(1, min(dx, dy))

    def _should_use_tile(self, dx, dy, exposure):
        if (dx, dy) not in self.tile_map:
            return False
        return max(exposure.values()) <= max(1, min(dx, dy))

    def _should_use_round_brick(self, dx, dy, exposure):
        if (dx, dy) != (1, 1):
            return False
        return sum(1 for value in exposure.values() if value > 0) >= 3

    def _quarter_turn_for_side(self, dx, dy, side):
        if dx >= dy:
            return {"front": 0, "left": 1, "back": 2, "right": 3}.get(side, 0)
        return {"left": 0, "back": 1, "right": 2, "front": 3}.get(side, 0)

    def _semantic_label_for_side(self, side):
        return {"front": 1, "right": 2, "back": 3, "left": 4}.get(side, 0)

    def _dominant_corner(self, exposure):
        ordered = sorted(exposure.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) < 2:
            return None
        side_a, side_b = ordered[0][0], ordered[1][0]
        if self._are_adjacent_sides(side_a, side_b):
            return tuple(sorted((side_a, side_b)))
        return None

    def _are_adjacent_sides(self, side_a, side_b):
        adjacent_pairs = {
            ("back", "left"),
            ("back", "right"),
            ("front", "left"),
            ("front", "right"),
        }
        return tuple(sorted((side_a, side_b))) in adjacent_pairs

    def _quarter_turn_for_corner(self, corner_pair):
        mapping = {
            ("front", "left"): 0,
            ("back", "left"): 1,
            ("back", "right"): 2,
            ("front", "right"): 3,
        }
        return mapping.get(corner_pair, 0)

    def _semantic_label_for_corner(self, corner_pair):
        mapping = {
            ("front", "left"): 5,
            ("back", "left"): 6,
            ("back", "right"): 7,
            ("front", "right"): 8,
        }
        return mapping.get(corner_pair, 0)
