from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import generate_binary_structure, label


def _connected_components(voxel_grid: np.ndarray):
    structure = generate_binary_structure(rank=3, connectivity=1)
    return label(voxel_grid > 0, structure=structure)


def compute_rule_risk_masks(voxel_grid: np.ndarray) -> Dict[str, np.ndarray]:
    vox = (voxel_grid > 0)
    shape = vox.shape
    empty_mask = np.zeros(shape, dtype=bool)
    if not vox.any():
        return {
            "floating_mask": empty_mask.copy(),
            "unsupported_mask": empty_mask.copy(),
            "isolated_mask": empty_mask.copy(),
            "rule_mask": empty_mask.copy(),
        }

    labeled, num = _connected_components(vox)
    ground_ids = np.unique(labeled[:, :, 0])
    ground_ids = ground_ids[ground_ids != 0]
    grounded = np.isin(labeled, ground_ids)
    floating_mask = vox & (~grounded)

    below_occupied = np.zeros_like(vox)
    below_occupied[:, :, 1:] = vox[:, :, :-1]
    unsupported_candidates = vox & (~below_occupied)

    lateral_neighbors = np.zeros_like(vox, dtype=np.int16)
    lateral_neighbors[1:, :, :] += vox[:-1, :, :]
    lateral_neighbors[:-1, :, :] += vox[1:, :, :]
    lateral_neighbors[:, 1:, :] += vox[:, :-1, :]
    lateral_neighbors[:, :-1, :] += vox[:, 1:, :]

    lower_ring_support = np.zeros_like(vox, dtype=np.int16)
    lower_ring_support[1:, :, 1:] += vox[:-1, :, :-1]
    lower_ring_support[:-1, :, 1:] += vox[1:, :, :-1]
    lower_ring_support[:, 1:, 1:] += vox[:, :-1, :-1]
    lower_ring_support[:, :-1, 1:] += vox[:, 1:, :-1]

    unsupported_mask = unsupported_candidates & (lateral_neighbors <= 1) & (lower_ring_support == 0)
    unsupported_mask[:, :, 0] = False
    unsupported_mask &= grounded

    isolated_mask = np.zeros_like(vox, dtype=bool)
    for cid in range(1, num + 1):
        component = labeled == cid
        comp_size = int(component.sum())
        touches_ground = cid in ground_ids
        if comp_size <= 2 and (not touches_ground):
            isolated_mask |= component

    rule_mask = floating_mask | unsupported_mask | isolated_mask
    return {
        "floating_mask": floating_mask,
        "unsupported_mask": unsupported_mask,
        "isolated_mask": isolated_mask,
        "rule_mask": rule_mask,
    }


def summarize_rule_risk(voxel_grid: np.ndarray) -> Dict[str, int]:
    masks = compute_rule_risk_masks(voxel_grid)
    rule_mask = masks["rule_mask"]
    floating_mask = masks["floating_mask"]
    unsupported_mask = masks["unsupported_mask"]
    isolated_mask = masks["isolated_mask"]
    labeled, num = _connected_components(rule_mask)
    return {
        "rule_risk_voxels": int(rule_mask.sum()),
        "floating_voxels": int(floating_mask.sum()),
        "unsupported_voxels": int(unsupported_mask.sum()),
        "isolated_voxels": int(isolated_mask.sum()),
        "rule_components": int(num),
    }


def bricks_overlapping_mask(brick_list: Sequence[dict], mask: np.ndarray) -> List[dict]:
    hits: List[dict] = []
    for brick in brick_list:
        grid_pos = brick.get("grid_pos")
        size = brick.get("size")
        if grid_pos is None or size is None:
            continue
        x, y, z = [int(v) for v in grid_pos]
        dx, dy = [int(v) for v in size]
        x1 = min(mask.shape[0], x + dx)
        y1 = min(mask.shape[1], y + dy)
        z1 = min(mask.shape[2], z + 1)
        if x < 0 or y < 0 or z < 0 or x >= x1 or y >= y1 or z >= z1:
            continue
        if mask[x:x1, y:y1, z:z1].any():
            hits.append(brick)
    return hits


def detect_risky_bricks(
    voxel_grid: np.ndarray,
    brick_list: Sequence[dict],
    risk_predictor=None,
    risk_threshold: float = 0.8,
) -> Tuple[List[dict], Dict[str, int], str]:
    rule_masks = compute_rule_risk_masks(voxel_grid)
    rule_stats = summarize_rule_risk(voxel_grid)
    rule_bricks = bricks_overlapping_mask(brick_list, rule_masks["rule_mask"])
    if rule_bricks:
        hints = []
        for brick in rule_bricks:
            hints.append(
                {
                    "id": brick.get("id"),
                    "grid_pos": brick.get("grid_pos"),
                    "size": brick.get("size"),
                    "risk_score": 1.0,
                    "reason": "rule_violation",
                }
            )
        return hints, rule_stats, "rules"

    if risk_predictor is None:
        return [], rule_stats, "none"

    predicted = risk_predictor.predict(brick_list, risk_threshold=risk_threshold)
    for item in predicted:
        item["reason"] = "gravity_proxy"
    return predicted, rule_stats, "predictor"
