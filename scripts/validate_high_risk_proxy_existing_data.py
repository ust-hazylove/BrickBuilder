import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.high_risk_predictor import HighRiskPredictor


TYPE_ID_TO_STRUCT = {
    2: "1x1",
    3: "1x1",
    5: "1x2",
    9: "2x4",
    10: "2x2",
    12: "2x3",
}

TYPE_ID_TO_SIZE = {
    2: (1, 1),
    3: (1, 1),
    5: (1, 2),
    9: (2, 4),
    10: (2, 2),
    12: (2, 3),
}

STYLE_VARIANTS = {
    "1x1": ["round_brick_1x1", "tile_1x1"],
    "1x2": ["grille_tile_1x2", "slope_1x2", "tile_1x2"],
    "2x2": ["corner_slope_2x2", "slope_2x2", "tile_2x2"],
}


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_score >= threshold
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def load_old_npz_sample(npz_path: Path) -> List[dict]:
    sample = np.load(npz_path, allow_pickle=False)
    pos = sample["pos"]
    type_idx = sample["type_idx"].astype(np.int64)
    rot = sample["rot"]
    semantic_label = sample["semantic_label"].astype(np.int64)

    bricks: List[dict] = []
    for idx in range(pos.shape[0]):
        type_id = int(type_idx[idx])
        struct_type = TYPE_ID_TO_STRUCT.get(type_id, "1x1")
        size = TYPE_ID_TO_SIZE.get(type_id, (1, 1))
        ori_quarter = int(round(float(rot[idx, 2]) / (math.pi / 2.0))) % 4
        z_layer = int(round(float(pos[idx, 2]) / 0.96))
        bricks.append(
            {
                "id": idx,
                "type_name": struct_type,
                "struct_type": struct_type,
                "size": [int(size[0]), int(size[1])],
                "grid_pos": [int(round(float(pos[idx, 0]))), int(round(float(pos[idx, 1]))), z_layer],
                "ori_quarter": ori_quarter,
                "semantic_label": int(semantic_label[idx]),
            }
        )
    return bricks


def predictor_scores(predictor: HighRiskPredictor, brick_list: Sequence[dict]) -> np.ndarray:
    if not brick_list:
        return np.zeros((0,), dtype=np.float32)
    node_features, pos, rot, edge_index, edge_attr = predictor._build_graph(brick_list)
    with torch.no_grad():
        stability, _ = predictor.model(node_features, pos, edge_index, edge_attr)
    risk = 1.0 - stability.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    return risk


def evaluate_old_distribution(
    predictor: HighRiskPredictor,
    data_root: Path,
    split: str,
    limit: int,
) -> Dict[str, object]:
    sample_paths = sorted((data_root / split).glob("*.npz"))
    if limit > 0:
        sample_paths = sample_paths[:limit]

    all_gt = []
    all_pred = []
    per_sample = []
    by_type_shift = Counter()

    for sample_path in sample_paths:
        sample = np.load(sample_path, allow_pickle=False)
        gt_stability = sample["V_i"].astype(np.float32).reshape(-1)
        gt_risk = 1.0 - gt_stability
        bricks = load_old_npz_sample(sample_path)
        pred_risk = predictor_scores(predictor, bricks)
        all_gt.append(gt_risk)
        all_pred.append(pred_risk)
        per_sample.append(
            {
                "sample_id": sample_path.stem,
                "num_bricks": int(len(bricks)),
                "gt_risk_mean": float(gt_risk.mean()),
                "pred_risk_mean": float(pred_risk.mean()),
                "corr": pearson_corr(gt_risk, pred_risk),
            }
        )
        for brick in bricks:
            by_type_shift[brick["struct_type"]] += 1

    gt = np.concatenate(all_gt, axis=0) if all_gt else np.zeros((0,), dtype=np.float32)
    pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,), dtype=np.float32)
    summary = {
        "num_samples": int(len(sample_paths)),
        "num_bricks": int(gt.shape[0]),
        "pearson_risk_corr": pearson_corr(gt, pred),
        "spearman_risk_corr": spearman_corr(gt, pred),
        "binary_gt_risk_lt_0_5": binary_metrics(gt >= 0.5, pred, threshold=0.5),
        "binary_gt_risk_lt_0_8": binary_metrics(gt >= 0.8, pred, threshold=0.8),
        "brick_type_counts": dict(sorted(by_type_shift.items())),
        "sample_preview": per_sample[:10],
    }
    return summary


def simulate_style_shift(
    predictor: HighRiskPredictor,
    data_root: Path,
    split: str,
    limit: int,
) -> Dict[str, object]:
    sample_paths = sorted((data_root / split).glob("*.npz"))
    if limit > 0:
        sample_paths = sample_paths[:limit]

    current_runtime_stats = defaultdict(list)
    hypothetical_type_sensitive_stats = defaultdict(list)
    coverage = Counter()

    for sample_path in sample_paths:
        base_bricks = load_old_npz_sample(sample_path)
        base_risk = predictor_scores(predictor, base_bricks)
        for src_type, styled_types in STYLE_VARIANTS.items():
            matching_ids = [i for i, brick in enumerate(base_bricks) if brick["struct_type"] == src_type]
            if not matching_ids:
                continue
            for styled_type in styled_types:
                variant_runtime = [dict(brick) for brick in base_bricks]
                for idx in matching_ids:
                    variant_runtime[idx]["type_name"] = styled_type
                runtime_risk = predictor_scores(predictor, variant_runtime)

                variant_sensitive = [dict(brick) for brick in base_bricks]
                for idx in matching_ids:
                    variant_sensitive[idx]["type_name"] = styled_type
                    variant_sensitive[idx]["struct_type"] = styled_type
                sensitive_risk = predictor_scores(predictor, variant_sensitive)

                coverage[styled_type] += len(matching_ids)
                runtime_delta = np.abs(runtime_risk - base_risk)
                sensitive_delta = np.abs(sensitive_risk - base_risk)
                current_runtime_stats[styled_type].append(
                    {
                        "mean_abs_delta": float(runtime_delta[matching_ids].mean()),
                        "mean_signed_delta": float((runtime_risk[matching_ids] - base_risk[matching_ids]).mean()),
                        "risk_ratio_gt_0_5_before": float((base_risk[matching_ids] >= 0.5).mean()),
                        "risk_ratio_gt_0_5_after": float((runtime_risk[matching_ids] >= 0.5).mean()),
                    }
                )
                hypothetical_type_sensitive_stats[styled_type].append(
                    {
                        "mean_abs_delta": float(sensitive_delta[matching_ids].mean()),
                        "mean_signed_delta": float((sensitive_risk[matching_ids] - base_risk[matching_ids]).mean()),
                        "risk_ratio_gt_0_5_before": float((base_risk[matching_ids] >= 0.5).mean()),
                        "risk_ratio_gt_0_5_after": float((sensitive_risk[matching_ids] >= 0.5).mean()),
                    }
                )

    def summarize_shift(rows_by_type):
        summary = {}
        for styled_type, rows in rows_by_type.items():
            mean_abs = np.mean([row["mean_abs_delta"] for row in rows]) if rows else 0.0
            mean_signed = np.mean([row["mean_signed_delta"] for row in rows]) if rows else 0.0
            before = np.mean([row["risk_ratio_gt_0_5_before"] for row in rows]) if rows else 0.0
            after = np.mean([row["risk_ratio_gt_0_5_after"] for row in rows]) if rows else 0.0
            summary[styled_type] = {
                "covered_bricks": int(coverage[styled_type]),
                "mean_abs_risk_shift": float(mean_abs),
                "mean_signed_risk_shift": float(mean_signed),
                "risk_ratio_before": float(before),
                "risk_ratio_after": float(after),
                "mapped_type_id": int(predictor.TYPE_TO_ID.get(styled_type, 2)),
            }
        return dict(sorted(summary.items()))

    return {
        "current_runtime_struct_type_priority": summarize_shift(current_runtime_stats),
        "hypothetical_if_predictor_saw_styled_type": summarize_shift(hypothetical_type_sensitive_stats),
    }


def analyze_new_library_outputs(
    predictor: HighRiskPredictor,
    dataset_root: Path,
    limit: int,
) -> Dict[str, object]:
    json_paths = sorted(dataset_root.glob("samples/*/target_bricks.json"))
    if limit > 0:
        json_paths = json_paths[:limit]

    type_counts = Counter()
    risky_counts = Counter()
    visual_type_counts = Counter()
    ignored_style_counts = Counter()
    sample_rows = []

    for json_path in json_paths:
        bricks = json.loads(json_path.read_text(encoding="utf-8"))
        if not bricks:
            continue
        risk = predictor_scores(predictor, bricks)
        for idx, brick in enumerate(bricks):
            effective_type = brick.get("struct_type", brick.get("type_name", "unknown"))
            visual_type = brick.get("type_name", effective_type)
            type_counts[effective_type] += 1
            visual_type_counts[visual_type] += 1
            if visual_type != effective_type:
                ignored_style_counts[visual_type] += 1
            if risk[idx] >= 0.5:
                risky_counts[effective_type] += 1
        sample_rows.append(
            {
                "sample_id": json_path.parent.name,
                "num_bricks": int(len(bricks)),
                "mean_risk": float(risk.mean()),
                "max_risk": float(risk.max()),
            }
        )

    per_type = {}
    for type_name, total in sorted(type_counts.items()):
        per_type[type_name] = {
            "count": int(total),
            "risky_count_at_0_5": int(risky_counts[type_name]),
            "risky_ratio_at_0_5": float(risky_counts[type_name] / max(total, 1)),
            "mapped_type_id": int(predictor.TYPE_TO_ID.get(type_name, 2)),
            "known_to_predictor": bool(type_name in predictor.TYPE_TO_ID),
        }

    return {
        "num_samples": int(len(sample_rows)),
        "visual_type_counts": dict(visual_type_counts.most_common(20)),
        "ignored_visual_style_counts": dict(ignored_style_counts.most_common(20)),
        "effective_type_analysis": per_type,
        "sample_preview": sample_rows[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate current high-risk proxy on existing old/new data.")
    parser.add_argument(
        "--checkpoint",
        default=r".\weights\high_risk_predictor_styled_best.pt",
    )
    parser.add_argument(
        "--stablelego_root",
        default=r"data/stablelego_50k/processed_contactpoint_refined",
    )
    parser.add_argument(
        "--lego_style_root",
        default=r"output\lego_style_dataset_v1",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--old_limit", type=int, default=200)
    parser.add_argument("--new_limit", type=int, default=200)
    parser.add_argument(
        "--output",
        default=r"output\high_risk_proxy_validation_existing_data\summary.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    predictor = HighRiskPredictor(checkpoint_path=args.checkpoint, device=args.device)
    old_summary = evaluate_old_distribution(
        predictor=predictor,
        data_root=Path(args.stablelego_root),
        split=args.split,
        limit=args.old_limit,
    )
    style_shift = simulate_style_shift(
        predictor=predictor,
        data_root=Path(args.stablelego_root),
        split=args.split,
        limit=args.old_limit,
    )
    new_library = analyze_new_library_outputs(
        predictor=predictor,
        dataset_root=Path(args.lego_style_root),
        limit=args.new_limit,
    )

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "old_distribution_eval": old_summary,
        "style_shift_simulation": style_shift,
        "new_library_output_analysis": new_library,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
