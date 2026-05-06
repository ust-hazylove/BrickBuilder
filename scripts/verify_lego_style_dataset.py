import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the LEGO-style finetune dataset bundle.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--check_samples", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    manifest_path = dataset_root / "manifest.jsonl"
    train_path = dataset_root / "train.txt"
    val_path = dataset_root / "val.txt"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing train.txt or val.txt")

    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    required_fields = [
        "id",
        "input_image",
        "target_voxels",
        "target_mesh",
        "target_render",
        "target_bricks",
        "split",
    ]
    checked = 0
    for row in rows[: args.check_samples]:
        for field in required_fields:
            if field not in row:
                raise KeyError(f"Missing field '{field}' in row: {row.get('id')}")
        for field in ["input_image", "target_voxels", "target_mesh", "target_render", "target_bricks"]:
            path = Path(row[field])
            if not path.exists():
                raise FileNotFoundError(f"Missing sample artifact: {path}")
        checked += 1

    train_count = len([line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    val_count = len([line for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    summary = {
        "dataset_root": str(dataset_root),
        "samples": len(rows),
        "checked_samples": checked,
        "train_ids": train_count,
        "val_ids": val_count,
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
