import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.high_risk_predictor import HighRiskPredictor
from modules.risk_analysis import detect_risky_bricks, summarize_rule_risk


def load_bricks(json_path: Path):
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Brick JSON must be a list of brick dicts.")
    return payload


def load_voxels(npz_path: Path):
    payload = np.load(npz_path, allow_pickle=False)
    if "voxels" in payload:
        return payload["voxels"]
    if "voxel_grid" in payload:
        return payload["voxel_grid"]
    keys = list(payload.keys())
    if len(keys) == 1:
        return payload[keys[0]]
    raise KeyError(f"Could not find voxel grid in npz. Keys: {keys}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone entry for the styled high-risk proxy.")
    parser.add_argument("--bricks_json", required=True, help="Path to brick list JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to predictor checkpoint.")
    parser.add_argument("--output_json", required=True, help="Where to save the result JSON.")
    parser.add_argument("--device", default="cpu", help="Device for PyTorch inference.")
    parser.add_argument("--risk_threshold", type=float, default=0.9, help="Risk threshold.")
    parser.add_argument("--voxels_npz", default=None, help="Optional voxel grid npz for rules + proxy detection.")
    args = parser.parse_args()

    bricks_path = Path(args.bricks_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output_json).resolve()
    voxels_path = Path(args.voxels_npz).resolve() if args.voxels_npz else None

    predictor = HighRiskPredictor(str(checkpoint_path), device=args.device)
    brick_list = load_bricks(bricks_path)

    if voxels_path is None:
        risky = predictor.predict(brick_list, risk_threshold=args.risk_threshold)
        result = {
            "mode": "predictor_only",
            "checkpoint": str(checkpoint_path),
            "brick_count": len(brick_list),
            "risk_threshold": float(args.risk_threshold),
            "risky_bricks": risky,
        }
    else:
        voxel_grid = load_voxels(voxels_path)
        risky, rule_stats, source = detect_risky_bricks(
            voxel_grid=voxel_grid,
            brick_list=brick_list,
            risk_predictor=predictor,
            risk_threshold=args.risk_threshold,
        )
        result = {
            "mode": "rules_plus_predictor",
            "checkpoint": str(checkpoint_path),
            "brick_count": len(brick_list),
            "risk_threshold": float(args.risk_threshold),
            "risk_source": source,
            "rule_stats": rule_stats,
            "direct_rule_summary": summarize_rule_risk(voxel_grid),
            "risky_bricks": risky,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
