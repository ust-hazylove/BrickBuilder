# config.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


# =========================
# Mask / Interface constraints
# =========================
@dataclass
class MaskCfg:
    # enable/disable hard constraints
    enable_T: bool = True
    enable_C: bool = True
    enable_I: bool = False   #暂时不开启库存
    enable_O: bool = True
    enable_S: bool = False   # 通常在 planner 里用老师判别器做严格 S，这里默认 False
    enable_U: bool = False   # 机械臂路径检查占位

    # O-approach (top-down insertion)
    approach: str = "top-down"   # "top-down" | "bottom-up" | "either"
    clearance: int = 1

    # S-threshold（仅当 enable_S=True 且未注入老师函数时使用）
    stability_thresh: float = 1.0


# =========================
# Clustering (layer-aware Louvain + postprocess)
# =========================
@dataclass
class ClusterCfg:
    alpha_cross: float = 0.9      # 跨层惩罚（越大越鼓励层内成团）
    max_outer_levels: int = 3
    max_local_passes: int = 6
    min_size: int = 3             # 小团合并阈值
    min_cohesion: float = 0.3     # 内聚力阈值（过低会被并入邻团）


# =========================
# Candidates (generation & prefilters)
# =========================
@dataclass
class CandidateCfg:
    # 通用
    max_candidates_per_step: int = 512
    anchor_stride: int = 1
    allow_orients: Tuple[int, ...] = (0, 1)  # 0:+X, 1:+Y

    # 物理预筛（candidates.py 内部的快速过滤）
    require_support_ratio_intra: float = 0.25
    require_support_ratio_inter: float = 0.20
    require_touch_existing: bool = True

    # 接口候选（子块对接）
    interface_band_halfwidth: float = 4.0
    interface_stride: int = 1

    # 锚点采样模式： "surface"（默认）或 "need"
    anchor_mode_intra: str = "surface"


# =========================
# Model / scoring & selection
# =========================
@dataclass
class ModelCfg:
    beam_size: int = 4
    # 启发式打分权重（seq_model.HeuristicScorer）
    w_voxels: float = 1.0
    w_need_covered: float = 1.2
    w_support_ratio: float = 0.8
    w_side_contacts: float = 0.2
    w_z_mean: float = 0.02       # 惩罚（更低更好）→ 实际用负号
    w_z_max: float = 0.01        # 惩罚
    w_risk_proxy: float = 0.6    # 惩罚（≈ 1 - support_ratio）


# =========================
# Planner (teacher-driven)
# 与 assemblr/core/planner.py 内的同名结构一致
# =========================
@dataclass
class TeacherPhysicsCfg:
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
class RiskCfg:
    use_mad_thresh: bool = True
    fixed_thresh: float = -1.0   # 无效（use_mad_thresh=True 时忽略）
    mad_k: float = 3.0

@dataclass
class PlanHeuristics:
    prefer_bottom_up: bool = True
    w_support: float = 1.0
    w_neg_risk_sum: float = 0.05
    w_neg_zmean: float = 0.01
    enable_soft_fallback: bool = True

@dataclass
class PlannerCfgGlue:
    physics: TeacherPhysicsCfg = TeacherPhysicsCfg()
    risk: RiskCfg = RiskCfg()
    heur: PlanHeuristics = PlanHeuristics()
    max_steps: int = 100000


# =========================
# Global bundle
# =========================
@dataclass
class Config:
    mask: MaskCfg = MaskCfg()
    cluster: ClusterCfg = ClusterCfg()
    cand: CandidateCfg = CandidateCfg()
    model: ModelCfg = ModelCfg()
    planner: PlannerCfgGlue = PlannerCfgGlue()


# 默认全局配置实例
CFG = Config()


# =========================
# Helper: build planner.PlannerCfg from this file
# =========================
def build_planner_cfg():
    """
    把本文件中的 PlannerCfgGlue 转为 assemblr.core.planner.PlannerCfg。
    如果你直接使用 run_plan.py 的命令行覆盖，这个函数可以不用。
    """
    try:
        # 延迟引入，避免纯导入 config.py 时的循环依赖
        from assemblr.core.planner import PlannerCfg as PPlannerCfg
        from assemblr.core.planner import TeacherPhysicsCfg as PTeacherPhysicsCfg
        from assemblr.core.planner import RiskCfg as PRiskCfg
        from assemblr.core.planner import PlanHeuristics as PPlanHeuristics
    except Exception:
        # 如果在非项目结构下使用，可直接返回 glue 对象
        return CFG.planner

    p = CFG.planner
    physics = PTeacherPhysicsCfg(
        cap_per_stud=p.physics.cap_per_stud,
        shear_cap=p.physics.shear_cap,
        mu_vert=p.physics.mu_vert,
        c0_vert=p.physics.c0_vert,
        mu_ground=p.physics.mu_ground,
        c0_ground=p.physics.c0_ground,
        alpha_reg=p.physics.alpha_reg,
        beta_reg=p.physics.beta_reg,
        ground_rigid=p.physics.ground_rigid,
        flip_y_physics=p.physics.flip_y_physics,
    )
    risk = PRiskCfg(
        use_mad_thresh=p.risk.use_mad_thresh,
        fixed_thresh=p.risk.fixed_thresh,
        mad_k=p.risk.mad_k,
    )
    heur = PPlanHeuristics(
        prefer_bottom_up=p.heur.prefer_bottom_up,
        w_support=p.heur.w_support,
        w_neg_risk_sum=p.heur.w_neg_risk_sum,
        w_neg_zmean=p.heur.w_neg_zmean,
        enable_soft_fallback=p.heur.enable_soft_fallback,
    )
    return PPlannerCfg(
        physics=physics,
        risk=risk,
        heur=heur,
        max_steps=p.max_steps,
    )
