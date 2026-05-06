# seq_model.py
# -*- coding: utf-8 -*-
"""
Lightweight scoring models for assembly candidate actions.

Two scorers:
- HeuristicScorer: physics-aware geometric features with configurable weights.
- TinyScorer: minimal baseline (voxels + small height preference).

Both expose:
    score(actions: List[Action], V_current: np.ndarray=None, V_target: np.ndarray=None) -> np.ndarray

Notes
- Actions are from interface_mask.Action (brick_type, anchor, orient, vox).
- V_current / V_target are 0/1 uint8 voxel grids.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    # optional: for type hints only; avoid hard import if not present
    from .interface_mask import Action  # noqa: F401
except Exception:
    # light fallback type for static checkers
    class Action:  # type: ignore
        def __init__(self, brick_type: str, anchor: Tuple[int, int, int], orient: int, vox: np.ndarray):
            self.brick_type = brick_type
            self.anchor = anchor
            self.orient = orient
            self.vox = vox


# ------------------------- utilities -------------------------
def _support_ratio(V_current: np.ndarray, add: np.ndarray) -> float:
    xs, ys, zs = np.where(add > 0)
    if xs.size == 0:
        return 0.0
    sup = 0
    for x, y, z in zip(xs, ys, zs):
        if z == 0 or V_current[x, y, z - 1] == 1:
            sup += 1
    return float(sup) / float(xs.size)

def _side_contacts(V_current: np.ndarray, add: np.ndarray) -> int:
    """
    Count 4-neighborhood side contacts with existing structure (no top/bottom).
    """
    xs, ys, zs = np.where(add > 0)
    if xs.size == 0:
        return 0
    nx, ny, nz = V_current.shape
    cnt = 0
    for x, y, z in zip(xs, ys, zs):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny:
                if V_current[xx, yy, z] == 1:
                    cnt += 1
    return int(cnt)

def _need_covered(V_current: np.ndarray, V_target: np.ndarray, add: np.ndarray) -> int:
    """
    Number of voxels that this action contributes to the target that are not yet present.
    """
    need = (V_target - V_current) > 0
    return int(np.sum((add > 0) & need))

def _z_stats(add: np.ndarray) -> Tuple[float, float]:
    xs, ys, zs = np.where(add > 0)
    if zs.size == 0:
        return 0.0, 0.0
    return float(np.mean(zs)), float(np.max(zs))

def _risk_proxy(add: np.ndarray, V_current: Optional[np.ndarray]=None) -> float:
    """
    Heuristic v_max proxy ~ 1 - support_ratio. If V_current missing, return 0.5.
    """
    if V_current is None:
        return 0.5
    r = _support_ratio(V_current, add)
    return float(max(0.0, 1.0 - r))


# ------------------------- Tiny baseline -------------------------
class TinyScorer:
    """
    Minimal scorer used in early MVP:
        score = voxels + 0.01 * z_mean (lower is better -> negative)
    """

    def score(
        self,
        actions: List[Action],
        V_current: Optional[np.ndarray] = None,
        V_target: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not actions:
            return np.zeros((0,), dtype=np.float32)
        scores = []
        for a in actions:
            add = a.vox
            voxels = int(np.sum(add))
            z_mean, _ = _z_stats(add)
            # prefer more voxels and slightly lower height
            s = voxels + 0.01 * (-z_mean)
            scores.append(float(s))
        return np.asarray(scores, dtype=np.float32)


# ------------------------- Heuristic scorer -------------------------
@dataclass
class HeuristicWeights:
    # positive benefits
    w_voxels: float = 1.0
    w_need_covered: float = 1.2
    w_support_ratio: float = 0.8
    w_side_contacts: float = 0.2
    # penalties (applied as negative)
    w_z_mean: float = 0.02
    w_z_max: float = 0.01
    w_risk_proxy: float = 0.6

class HeuristicScorer:
    """
    Physics-aware geometric features with configurable weights.
    Intended as a fast pre-ranking prior to teacher evaluation.

    score =
        + w_voxels        * voxels
        + w_need_covered  * need_covered
        + w_support_ratio * support_ratio
        + w_side_contacts * side_contacts
        - w_z_mean        * z_mean
        - w_z_max         * z_max
        - w_risk_proxy    * risk_proxy   (≈ 1 - support_ratio)

    All terms are computed per-candidate independently (no global normalization).
    """

    def __init__(self, w: HeuristicWeights = HeuristicWeights()):
        self.w = w

    def _feat_vector(self, a: Action, V_current: np.ndarray, V_target: np.ndarray) -> Tuple[float, ...]:
        add = a.vox
        voxels = float(np.sum(add))
        need_cov = float(_need_covered(V_current, V_target, add))
        sup = float(_support_ratio(V_current, add))
        side = float(_side_contacts(V_current, add))
        z_mean, z_max = _z_stats(add)
        risk_p = float(_risk_proxy(add, V_current))
        return (voxels, need_cov, sup, side, z_mean, z_max, risk_p)

    def score(
        self,
        actions: List[Action],
        V_current: Optional[np.ndarray],
        V_target: Optional[np.ndarray]
    ) -> np.ndarray:
        if not actions:
            return np.zeros((0,), dtype=np.float32)
        if V_current is None or V_target is None:
            raise ValueError("HeuristicScorer requires V_current and V_target.")

        W = self.w
        out = np.empty((len(actions),), dtype=np.float32)
        for i, a in enumerate(actions):
            voxels, need_cov, sup, side, z_mean, z_max, risk_p = self._feat_vector(a, V_current, V_target)
            s = (
                W.w_voxels * voxels +
                W.w_need_covered * need_cov +
                W.w_support_ratio * sup +
                W.w_side_contacts * side -
                W.w_z_mean * z_mean -
                W.w_z_max * z_max -
                W.w_risk_proxy * risk_p
            )
            out[i] = float(s)
        return out


# ------------------------- selection helpers -------------------------
def masked_topk(scores: np.ndarray, mask_bool: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Select top-k indices under a boolean mask. Returns indices (np.ndarray).
    """
    if scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    masked = np.where(mask_bool, scores, -1e18)
    idx = np.argsort(-masked)[:k]
    return idx

def masked_argmax(scores: np.ndarray, mask_bool: np.ndarray, beam: int = 1) -> List[int]:
    """
    Backward compatible helper (used by older code).
    """
    return masked_topk(scores, mask_bool, k=max(1, beam)).tolist()