# planner.py
# -*- coding: utf-8 -*-
"""
Assembly sequence planner driven by the teacher stability solver (ldr_stability.py).

Goal:
  Given a target brick list (with fixed poses) describing a valid LDraw design,
  produce a physically executable assembly order such that every prefix is stable
  under the teacher's physics model.

Key ideas:
  - At each step, only consider bricks not yet placed.
  - Tentatively add one candidate brick; run the teacher solver to evaluate stability.
  - Accept candidates whose post-state max risk < adaptive threshold (MAD rule).
  - Among acceptable candidates, prefer the one that maximizes vertical support gain
    (number of new vertical contacts introduced by the brick). Tie-break with minimal
    global risk sum, then lower height (bottom-up).
  - If no candidate passes, try a soft fallback: pick the brick that yields the
    smallest max-risk; if still > threshold, report deadlock (design may need
    temporary subassembly or fixturing).

You can plug this module directly into your pipeline where target bricks are
parsed from LDR (same coordinates/orientation). Uses ONLY teacher’s rules.

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import copy
import numpy as np

# === Import teacher solver primitives ===
# NOTE: We call into your teacher exactly as implemented.
from Experiments.ldr_stability import (
    Brick as TBrick,
    apply_y_flip_in_physics,
    build_world_grid,
    build_and_solve,
)

# ------------- Config dataclasses -------------
@dataclass
class TeacherPhysicsCfg:
    # Physics/caps (exactly the same semantics as ldr_stability.py)
    cap_per_stud: float = 12.0
    shear_cap: float = 4.0
    mu_vert: float = 0.35
    c0_vert: float = 0.25
    mu_ground: float = 0.45
    c0_ground: float = 0.50
    alpha_reg: float = 0.0
    beta_reg: float = 0.0
    ground_rigid: bool = False
    flip_y_physics: bool = True

@dataclass
class PlanHeuristics:
    # prefer bottom-up
    prefer_bottom_up: bool = True
    # weight for selecting candidate: (support_gain, -risk_sum, -z_mean)
    w_support: float = 1.0
    w_neg_risk_sum: float = 0.05
    w_neg_zmean: float = 0.01
    # if no candidate passes threshold, allow soft fallback
    enable_soft_fallback: bool = True

@dataclass
class RiskCfg:
    # Adaptive threshold via MAD (same rule as teacher process_one)
    use_mad_thresh: bool = True
    fixed_thresh: float = -1.0  # <0 ignored when use_mad_thresh==True
    mad_k: float = 3.0

@dataclass
class PlannerCfg:
    physics: TeacherPhysicsCfg = TeacherPhysicsCfg()
    risk: RiskCfg = RiskCfg()
    heur: PlanHeuristics = PlanHeuristics()
    # safety bounds
    max_steps: int = 100000


# ------------- Utility: adaptive risk threshold -------------
def _mad_threshold(risk: np.ndarray, k: float = 3.0) -> float:
    """Median +/- k*MAD rule (same as in teacher)."""
    med = float(np.median(risk))
    mad = float(np.median(np.abs(risk - med)) + 1e-9)
    return med + k * 1.4826 * mad


# ------------- Core evaluation against teacher -------------
def _eval_teacher(
    bricks: List[TBrick],
    cfg: TeacherPhysicsCfg,
    *,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run the teacher solver on a *copied* brick list.
    Returns:
      risk: np.ndarray (per-brick)
      info: dict with brick_stats / vert_stats / horiz_stats / occ, etc.
    Notes:
      - We mirror Y in PHYSICS if cfg.flip_y_physics is True (same as teacher default).
      - Contacts/world grid always rebuilt from bricks (teacher is brick-level, no vox).
    """
    # Deepcopy bricks so we can safely flip-Y in physics without affecting caller
    bricks_c = copy.deepcopy(bricks)

    # Apply PHYSICS Y-flip exactly like teacher pipeline (default ON).
    apply_y_flip_in_physics(bricks_c, enable=cfg.flip_y_physics)

    # Build world + solve
    occ_half, horiz_slices, vert_pairs = build_world_grid(bricks_c)
    risk, brick_stats, vert_stats, horiz_stats = build_and_solve(
        bricks_c, occ_half, horiz_slices, vert_pairs,
        cfg.cap_per_stud, cfg.shear_cap,
        cfg.mu_vert, cfg.c0_vert, cfg.mu_ground, cfg.c0_ground,
        cfg.alpha_reg, cfg.beta_reg,
        ground_rigid=cfg.ground_rigid,
        verbose=verbose
    )
    info = {
        "brick_stats": brick_stats,
        "vert_stats": vert_stats,
        "horiz_stats": horiz_stats,
        "occ_half": occ_half,
        "horiz_slices": horiz_slices,
        "vert_pairs": vert_pairs,
        "bricks_used": bricks_c,  # NOTE: y-flipped copy used in physics
    }
    return risk, info


def _count_new_vertical_supports(
    info_before: Dict[str, Any],
    info_after: Dict[str, Any],
    new_brick_idx_after: int
) -> int:
    """
    Count how many vertical contact 'pairs' the newly placed brick participates in
    after the action, minus what it had before (usually 0 since it didn't exist).
    """
    def _collect_pairs(vert_stats: List[Dict[str, Any]], bid: int) -> int:
        cnt = 0
        for s in vert_stats:
            up = int(s.get("upper_brick", -999))
            dn = int(s.get("lower_brick", -999))
            if up == bid or dn == bid:
                cnt += 1
        return cnt

    before_cnt = _collect_pairs(info_before.get("vert_stats", []), new_brick_idx_after)
    after_cnt = _collect_pairs(info_after.get("vert_stats", []), new_brick_idx_after)
    return max(0, after_cnt - before_cnt)


def _z_mean_of_brick(b: TBrick) -> float:
    """Use Y as height in teacher; here for tie-breaking we favor lower bricks first."""
    return float(b.y0_b + 0.5 * b.h_b)


# ------------- The Planner -------------
class TeacherDrivenPlanner:
    """
    Produce a stable assembly order (list of brick indices) for a GIVEN target brick list.
    We assume target bricks are collision-free and describe the final geometry/pose.
    """

    def __init__(self, cfg: PlannerCfg = PlannerCfg()):
        self.cfg = cfg

    def plan(self, target_bricks: List[TBrick], *, verbose: bool = False) -> Dict[str, Any]:
        """
        Returns:
          dict:
            order: List[int]             # indices (in target list) in assembly order
            steps: List[Dict]            # logs per step
            done: bool
            reason: str
            final_risk: np.ndarray
        """
        N = len(target_bricks)
        if N == 0:
            return dict(order=[], steps=[], done=True, reason="empty", final_risk=np.zeros((0,)))

        # Book-keeping: which target indices are already placed
        placed_mask = np.zeros(N, dtype=bool)
        order: List[int] = []
        steps_log: List[Dict[str, Any]] = []

        # Current prefix of bricks (using the SAME dataclass as teacher)
        bricks_cur: List[TBrick] = []

        # Evaluate empty world once
        risk0, info0 = _eval_teacher(bricks_cur, self.cfg.physics, verbose=verbose)
        # Empty risk array -> use small placeholder
        cur_risk = np.asarray(risk0 if len(risk0) else np.zeros((0,), dtype=float))
        cur_info = info0

        # Loop
        max_steps = min(self.cfg.max_steps, N)
        for t in range(max_steps):
            remain = [i for i in range(N) if not placed_mask[i]]
            if not remain:
                # done
                final_risk, _ = _eval_teacher(bricks_cur, self.cfg.physics, verbose=verbose)
                return dict(order=order, steps=steps_log, done=True, reason="all_placed", final_risk=final_risk)

            # Adaptive threshold on current risk landscape (teacher rule)
            if self.cfg.risk.use_mad_thresh:
                # if current prefix has no bricks, fallback to tiny positive threshold
                cur_thresh = _mad_threshold(cur_risk, k=self.cfg.risk.mad_k) if cur_risk.size > 0 else 1e-6
            else:
                cur_thresh = self.cfg.risk.fixed_thresh if self.cfg.risk.fixed_thresh >= 0 else 1.0

            candidate_scores: List[Tuple[float, int, Dict[str, Any]]] = []
            fallback_records: List[Tuple[float, int, Dict[str, Any]]] = []

            # Evaluate each remaining brick as "next action"
            for idx in remain:
                b_new: TBrick = target_bricks[idx]

                # Create tentative state = placed + new brick
                tmp = bricks_cur + [b_new]
                # Evaluate teacher for tentative state
                risk_after, info_after = _eval_teacher(tmp, self.cfg.physics, verbose=False)
                v_max = float(np.max(risk_after)) if risk_after.size > 0 else 0.0

                # Compute adaptive threshold based on tentative state's distribution
                if self.cfg.risk.use_mad_thresh:
                    v_thresh = _mad_threshold(risk_after, k=self.cfg.risk.mad_k)
                else:
                    v_thresh = cur_thresh

                # Support gain: how many new vertical contact pairs the new brick made
                # NOTE: teacher indices in info_after correspond to order within 'tmp'
                # new brick index in the 'tmp' world is len(tmp)-1
                support_gain = _count_new_vertical_supports(cur_info, info_after, new_brick_idx_after=len(tmp)-1)

                # Tie-breakers
                zmean = _z_mean_of_brick(b_new)
                risk_sum = float(np.sum(risk_after))

                # Acceptable?
                acceptable = (v_max < v_thresh)

                rec = dict(
                    idx=idx,
                    v_max=v_max,
                    v_thresh=v_thresh,
                    support_gain=support_gain,
                    risk_sum=risk_sum,
                    z_mean=zmean,
                    info_after=info_after,
                    risk_after=risk_after,
                )

                # Accumulate fallback list (min v_max) regardless of acceptable or not
                fallback_records.append((v_max, idx, rec))

                if acceptable:
                    score = (
                        self.cfg.heur.w_support * support_gain
                        + self.cfg.heur.w_neg_risk_sum * (-risk_sum)
                        + self.cfg.heur.w_neg_zmean * (-zmean if self.cfg.heur.prefer_bottom_up else 0.0)
                    )
                    candidate_scores.append((score, idx, rec))

            # Pick best acceptable candidate
            if candidate_scores:
                candidate_scores.sort(key=lambda t3: (-t3[0], t3[1]))   # max score, then smaller idx
                _, chosen_idx, chosen = candidate_scores[0]
                # Commit
                bricks_cur.append(target_bricks[chosen_idx])
                placed_mask[chosen_idx] = True
                order.append(chosen_idx)
                cur_risk = chosen["risk_after"]
                cur_info = chosen["info_after"]

                steps_log.append(dict(
                    step=t,
                    chosen=chosen_idx,
                    v_max=chosen["v_max"],
                    v_thresh=chosen["v_thresh"],
                    support_gain=chosen["support_gain"],
                    risk_sum=chosen["risk_sum"],
                    z_mean=chosen["z_mean"],
                    mode="accept",
                ))
                continue

            # No acceptable candidate -> try soft fallback
            if self.cfg.heur.enable_soft_fallback and fallback_records:
                fallback_records.sort(key=lambda t3: (t3[0], t3[1]))  # min v_max
                v_min, chosen_idx, chosen = fallback_records[0]
                # If even the smallest v_max is "close" to threshold, we can try committing
                # Here we use a mild slack (10%) to avoid deadlock on marginal cases
                slack_ok = (v_min < 1.10 * chosen["v_thresh"])
                if slack_ok:
                    bricks_cur.append(target_bricks[chosen_idx])
                    placed_mask[chosen_idx] = True
                    order.append(chosen_idx)
                    cur_risk = chosen["risk_after"]
                    cur_info = chosen["info_after"]

                    steps_log.append(dict(
                        step=t,
                        chosen=chosen_idx,
                        v_max=chosen["v_max"],
                        v_thresh=chosen["v_thresh"],
                        support_gain=chosen["support_gain"],
                        risk_sum=chosen["risk_sum"],
                        z_mean=chosen["z_mean"],
                        mode="soft-fallback",
                    ))
                    continue

            # Deadlock: no brick keeps the structure under threshold
            final_risk, _ = _eval_teacher(bricks_cur, self.cfg.physics, verbose=verbose)
            return dict(
                order=order, steps=steps_log, done=False,
                reason="deadlock_no_acceptable_candidate",
                final_risk=final_risk
            )

        # Safety exit (shouldn't happen for valid N)
        final_risk, _ = _eval_teacher(bricks_cur, self.cfg.physics, verbose=verbose)
        return dict(order=order, steps=steps_log, done=False, reason="max_steps_reached", final_risk=final_risk)


# ------------- Minimal usage example -------------
if __name__ == "__main__":
    # Example: suppose you have already parsed LDR into a list[TBrick] named `target_bricks`
    # using your existing ldr parser (same as ldr_stability.parse_ldr_to_bricks).
    #
    # from ldr_stability import parse_ldr_to_bricks
    # target_bricks = parse_ldr_to_bricks("path/to/model.ldr", y_relax_deg=6.0,
    #                                     force_upright=False, accept_inverted=True,
    #                                     keep_unknown=True, verbose=True)
    #
    # Then:
    cfg = PlannerCfg()
    planner = TeacherDrivenPlanner(cfg)
    # result = planner.plan(target_bricks, verbose=False)
    # print("DONE:", result["done"], "reason:", result["reason"])
    # print("ORDER:", result["order"])
    pass