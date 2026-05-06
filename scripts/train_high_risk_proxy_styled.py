import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.high_risk_predictor import HighRiskPredictor, TAGUNetRiskPredictor


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_weights(counter: Counter) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in counter.items()}


def load_style_distribution(lego_style_root: Path, limit: int = 0) -> Dict[str, Dict[str, float]]:
    json_paths = sorted(lego_style_root.glob("samples/*/target_bricks.json"))
    if limit > 0:
        json_paths = json_paths[:limit]

    base_to_visual = defaultdict(Counter)
    for json_path in tqdm(json_paths, desc="Scanning style distribution", leave=False):
        bricks = json.loads(json_path.read_text(encoding="utf-8"))
        for brick in bricks:
            base = brick.get("struct_type") or brick.get("type_name")
            visual = brick.get("type_name") or base
            if base is None or visual is None:
                continue
            base_to_visual[base][visual] += 1

    distribution = {base: normalize_weights(counter) for base, counter in base_to_visual.items()}
    return distribution


def sample_visual_type(base_type: str, distribution: Dict[str, Dict[str, float]], rng: random.Random) -> str:
    probs = distribution.get(base_type)
    if not probs:
        return base_type
    labels = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def load_npz_as_bricks(npz_path: Path, style_distribution: Dict[str, Dict[str, float]], rng: random.Random) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    sample = np.load(npz_path, allow_pickle=False)
    pos = sample["pos"]
    type_idx = sample["type_idx"].astype(np.int64)
    rot = sample["rot"]
    semantic_label = sample["semantic_label"].astype(np.int64)
    stability = sample["V_i"].astype(np.float32).reshape(-1)
    stress = sample["F"].astype(np.float32)

    bricks: List[dict] = []
    for idx in range(pos.shape[0]):
        type_id = int(type_idx[idx])
        struct_type = TYPE_ID_TO_STRUCT.get(type_id, "1x1")
        size = TYPE_ID_TO_SIZE.get(type_id, (1, 1))
        visual_type = sample_visual_type(struct_type, style_distribution, rng)
        ori_quarter = int(round(float(rot[idx, 2]) / (np.pi / 2.0))) % 4
        z_layer = int(round(float(pos[idx, 2]) / 0.96))
        bricks.append(
            {
                "id": idx,
                "type_name": visual_type,
                "struct_type": struct_type,
                "size": [int(size[0]), int(size[1])],
                "grid_pos": [int(round(float(pos[idx, 0]))), int(round(float(pos[idx, 1]))), z_layer],
                "ori_quarter": ori_quarter,
                "semantic_label": int(semantic_label[idx]),
            }
        )
    return bricks, stability, stress


def iterate_epoch(
    sample_paths: Sequence[Path],
    predictor_wrapper: HighRiskPredictor,
    model: TAGUNetRiskPredictor,
    optimizer: torch.optim.Optimizer,
    style_distribution: Dict[str, Dict[str, float]],
    rng: random.Random,
    train: bool,
) -> Dict[str, float]:
    device = predictor_wrapper.device
    total_loss = 0.0
    total_bce = 0.0
    total_stress = 0.0
    total_nodes = 0

    if train:
        model.train()
    else:
        model.eval()

    iterator = tqdm(sample_paths, desc="train" if train else "val", leave=False)
    for sample_path in iterator:
        bricks, stability_target, stress_target = load_npz_as_bricks(sample_path, style_distribution, rng)
        node_features, pos, _, edge_index, edge_attr = predictor_wrapper._build_graph(bricks)
        target_stability = torch.tensor(stability_target, dtype=torch.float32, device=device)
        target_stress = torch.tensor(stress_target, dtype=torch.float32, device=device)

        with torch.set_grad_enabled(train):
            pred_stability, pred_stress = model(node_features, pos, edge_index, edge_attr)
            pred_stability = pred_stability.view(-1)
            bce = F.binary_cross_entropy(pred_stability, target_stability.clamp(0.0, 1.0))
            stress_loss = F.smooth_l1_loss(pred_stress, target_stress)
            loss = bce + 0.05 * stress_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        node_count = int(target_stability.numel())
        total_loss += float(loss.detach().cpu()) * node_count
        total_bce += float(bce.detach().cpu()) * node_count
        total_stress += float(stress_loss.detach().cpu()) * node_count
        total_nodes += node_count

    denom = max(total_nodes, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "stress": total_stress / denom,
        "nodes": total_nodes,
    }


def save_checkpoint(
    output_dir: Path,
    model: TAGUNetRiskPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    metrics: Dict[str, float],
    style_distribution: Dict[str, Dict[str, float]],
    name: str,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "physics_proxy": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "type_to_id": HighRiskPredictor.TYPE_TO_ID,
            "style_distribution": style_distribution,
        },
    }
    torch.save(payload, checkpoint_dir / name)


def append_log(csv_path: Path, row: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a styled high-risk proxy on runtime graph features.")
    parser.add_argument("--stablelego_root", default="data/stablelego_50k/processed_contactpoint_refined")
    parser.add_argument("--lego_style_root", default="output/lego_style_dataset_v1")
    parser.add_argument("--output_dir", default="output/styled_high_risk_proxy_run1")
    parser.add_argument("--base_checkpoint", default="weights/high_risk_predictor_styled_best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--style_scan_limit", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_distribution = load_style_distribution(Path(args.lego_style_root), limit=args.style_scan_limit)
    (output_dir / "style_distribution.json").write_text(
        json.dumps(style_distribution, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_paths = sorted((Path(args.stablelego_root) / "train").glob("*.npz"))
    val_paths = sorted((Path(args.stablelego_root) / "val").glob("*.npz"))
    if args.train_limit > 0:
        train_paths = train_paths[:args.train_limit]
    if args.val_limit > 0:
        val_paths = val_paths[:args.val_limit]

    predictor_wrapper = HighRiskPredictor(checkpoint_path=args.base_checkpoint, device=args.device)
    model = TAGUNetRiskPredictor(in_channels=128, hidden_channels=128, depth=3, edge_attr_dim=14).to(predictor_wrapper.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    log_path = output_dir / "train_log.csv"

    for epoch in range(1, args.epochs + 1):
        train_rng = random.Random(args.seed + epoch)
        val_rng = random.Random(args.seed + 10_000 + epoch)
        random.shuffle(train_paths)

        train_metrics = iterate_epoch(
            sample_paths=train_paths,
            predictor_wrapper=predictor_wrapper,
            model=model,
            optimizer=optimizer,
            style_distribution=style_distribution,
            rng=train_rng,
            train=True,
        )
        val_metrics = iterate_epoch(
            sample_paths=val_paths,
            predictor_wrapper=predictor_wrapper,
            model=model,
            optimizer=optimizer,
            style_distribution=style_distribution,
            rng=val_rng,
            train=False,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_stress": train_metrics["stress"],
            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_stress": val_metrics["stress"],
            "train_nodes": train_metrics["nodes"],
            "val_nodes": val_metrics["nodes"],
        }
        append_log(log_path, row)
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            metrics=row,
            style_distribution=style_distribution,
            name="latest.pt",
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metrics=row,
                style_distribution=style_distribution,
                name="best.pt",
            )

        print(json.dumps(row, ensure_ascii=False))

    print(f"Training complete. Logs: {log_path}")
    print(f"Best checkpoint: {output_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    main()
