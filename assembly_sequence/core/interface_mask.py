# interface_mask.py
# -*- coding: utf-8 -*-
"""
Physics-aware action masking for assembly planning.

包含的硬约束（可独立开关）：
- T 任务一致：新增体素全部落在目标体素内
- C 无碰撞：新增体素与当前结构不重叠
- I 库存可用：库存里有该砖
- O 可操作：装配路径可达（默认：自上而下插装，顶部留空；也可设置上下任一方向）
- S 稳定性：放置后结构的最大风险值 v_max < 阈值（可注入老师判别器）
- U 机械臂可操作（占位；需要时可接你的运动学/碰撞库）

使用方式：
1) （可选）注入稳定性函数：
    from interface_mask import set_stability_fn
    set_stability_fn(your_fn)  # your_fn(V_after: np.ndarray) -> float(v_max)
   其中 V_after 为放置后的体素占据（uint8 0/1）。
   如果你想直接调用 ldr_stability.py，可在外层把砖级场景→体素近似后传入；
   或在 planner.py 中直接调用老师求解（我们已这么做），此处 S 可关闭以节省时间。

2) 单步判定：
    ok = mask_action(action, V_cur, V_tgt, inventory, use_S=True, stability_thresh=1.0)

3) 批量判定：
    mask = mask_actions(actions, V_cur, V_tgt, inventory, use_S=False)

注意：
- 本文件不依赖老师求解器；S 的严格校验建议放在 planner（已接入你的 ldr_stability）以避免重复耗时。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable, Optional
import numpy as np


# ------------------------- 动作结构 -------------------------
@dataclass
class Action:
    """
    一个候选装配动作：
    - brick_type: 如 "1x2"、"2x4"
    - anchor: (x,y,z) 插装定位的体素起点（与 candidates.py 一致）
    - orient: 0/1 等朝向（与 candidates.py 一致）
    - vox: 新增体素，占据体 (0/1, shape=grid)
    """
    brick_type: str
    anchor: Tuple[int, int, int]
    orient: int
    vox: np.ndarray  # uint8 0/1 same grid as V_current / V_target


# ------------------------- 可注入的稳定性函数 -------------------------
_STABILITY_FN: Optional[Callable[[np.ndarray], float]] = None
# 期望签名：fn(V_after: np.ndarray) -> float(v_max)，返回“越小越稳”的最大风险值

def set_stability_fn(fn: Optional[Callable[[np.ndarray], float]]) -> None:
    """外部注入稳定性计算函数（例如来自你的老师判别器的体素近似版）"""
    global _STABILITY_FN
    _STABILITY_FN = fn


# ------------------------- T/C/I 基本几何与库存 -------------------------
def check_task_inside_target(add_voxels: np.ndarray, V_target: np.ndarray) -> bool:
    """T: 新增体素必须全部位于目标体素内"""
    return bool(np.all(add_voxels <= V_target))

def check_collision(add_voxels: np.ndarray, V_current: np.ndarray) -> bool:
    """C: 新增体素与当前结构不重叠"""
    return bool(np.all((add_voxels + V_current) <= 1))

def check_inventory_ok(inventory: Dict[str, int], brick_type: str) -> bool:
    """I: 库存中该砖型数量 > 0"""
    return inventory.get(brick_type, 0) > 0


# ------------------------- O 可操作性（插装路径） -------------------------
def _path_clear_top_down(add_voxels: np.ndarray, V_current: np.ndarray, clearance: int = 1) -> bool:
    """
    简化的“自上而下”插装可达性：新增体素上方至少有 `clearance` 层为空，
    即对 add 的每个体素 (x,y,z)，要求 V_current[x,y,z+1:z+1+clearance] 都为 0（边界外视为通畅）。
    """
    nx, ny, nz = V_current.shape
    xs, ys, zs = np.where(add_voxels > 0)
    if xs.size == 0:
        return False
    for x, y, z in zip(xs, ys, zs):
        z1 = z + 1
        z2 = min(nz, z + 1 + max(0, clearance))
        if z1 < nz:
            if np.any(V_current[x, y, z1:z2] > 0):
                return False
        # 边界外（z1>=nz）视为通畅
    return True

def _path_clear_bottom_up(add_voxels: np.ndarray, V_current: np.ndarray, clearance: int = 1) -> bool:
    """
    备用：自下而上插装可达性（一般不用）。对称定义。
    """
    nx, ny, nz = V_current.shape
    xs, ys, zs = np.where(add_voxels > 0)
    if xs.size == 0:
        return False
    for x, y, z in zip(xs, ys, zs):
        z2 = z
        z1 = max(0, z - clearance)
        if z1 < z2:
            if np.any(V_current[x, y, z1:z2] > 0):
                return False
    return True

def check_operable(
    add_voxels: np.ndarray,
    V_current: np.ndarray,
    *,
    approach: str = "top-down",   # "top-down" | "bottom-up" | "either"
    clearance: int = 1
) -> bool:
    """O: 插装可达性检查（简化）"""
    if approach == "top-down":
        return _path_clear_top_down(add_voxels, V_current, clearance=clearance)
    elif approach == "bottom-up":
        return _path_clear_bottom_up(add_voxels, V_current, clearance=clearance)
    else:
        return (
            _path_clear_top_down(add_voxels, V_current, clearance=clearance) or
            _path_clear_bottom_up(add_voxels, V_current, clearance=clearance)
        )


# ------------------------- S 稳定性（可注入 / 启发式） -------------------------
def _heuristic_vmax(V_after: np.ndarray) -> float:
    """
    轻量启发式 v_max：越多“有底支撑”的占比越大 → 越稳（v_max 越小）。
    这是一个占位近似，建议在 planner 里用老师判别器做严格验收。
    """
    xs, ys, zs = np.where(V_after > 0)
    if xs.size == 0:
        return 0.0
    nx, ny, nz = V_after.shape
    supported = 0
    for x, y, z in zip(xs, ys, zs):
        if z == 0 or (z > 0 and V_after[x, y, z - 1] > 0):
            supported += 1
    support_ratio = supported / float(xs.size)  # [0,1]
    # 将“越稳越小”的想法映射为 v_max≈1 - 支撑率（仅占位用）
    return float(max(0.0, 1.0 - support_ratio))

def eval_stability_score(V_after: np.ndarray) -> float:
    """
    返回最大风险值 v_max（越小越稳）。若已注入 _STABILITY_FN 则用之，否则用启发式。
    """
    if _STABILITY_FN is not None:
        try:
            return float(_STABILITY_FN(V_after))
        except Exception:
            # 兜底：避免注入函数异常导致流程中断
            return _heuristic_vmax(V_after)
    return _heuristic_vmax(V_after)


# ------------------------- U 机械臂可操作（占位） -------------------------
def check_robot_manipulability(candidate_pose) -> bool:
    """
    机械臂可达/无碰撞的占位检查。需要时替换为你的运动学/规划器调用。
    candidate_pose: (anchor, orient) 或更丰富的姿态描述
    """
    return True


# ------------------------- 单步与批量 Mask -------------------------
def mask_action(
    action: Action,
    V_current: np.ndarray,
    V_target: np.ndarray,
    inventory: Dict[str, int],
    *,
    # 开关
    use_T: bool = True,
    use_C: bool = True,
    use_I: bool = True,
    use_O: bool = True,
    use_S: bool = True,
    use_U: bool = False,
    # O 参数
    approach: str = "top-down",
    clearance: int = 1,
    # S 参数
    stability_thresh: float = 1.0,
) -> bool:
    """
    返回该动作是否通过所有启用的硬约束。
    """
    add = action.vox

    if use_T and not check_task_inside_target(add, V_target):
        return False
    if use_C and not check_collision(add, V_current):
        return False
    if use_I and not check_inventory_ok(inventory, action.brick_type):
        return False
    if use_O and not check_operable(add, V_current, approach=approach, clearance=clearance):
        return False

    if use_S:
        V_after = np.clip(V_current + add, 0, 1).astype(V_current.dtype)
        v_max = eval_stability_score(V_after)
        if v_max >= stability_thresh:
            return False

    if use_U:
        if not check_robot_manipulability((action.anchor, action.orient)):
            return False

    return True


def mask_actions(
    actions: List[Action],
    V_current: np.ndarray,
    V_target: np.ndarray,
    inventory: Dict[str, int],
    **kwargs
) -> np.ndarray:
    """
    批量版本：返回布尔数组（len(actions),）。
    kwargs 与 mask_action 的命名参数一致。
    """
    if not actions:
        return np.zeros((0,), dtype=bool)
    m = []
    for a in actions:
        ok = mask_action(a, V_current, V_target, inventory, **kwargs)
        m.append(bool(ok))
    return np.asarray(m, dtype=bool)