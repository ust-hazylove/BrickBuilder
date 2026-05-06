# Reviewer Guide

This document explains how to inspect, run, and reproduce the anonymous Img2Build/Diff-LEGO artifact.

## 1. Method overview

The implemented pipeline is geometry-first rather than direct text/image-to-bricks. The main path is:

1. Input image is processed by Hunyuan3D-2.1 shape generation.
2. The generated mesh is normalized and voxelized to a capped grid resolution, normally `16^3` for review/demo runs.
3. Rule-based risk analysis detects floating, unsupported, and isolated voxels.
4. A learned high-risk proxy optionally flags risky brick regions.
5. A PPO repair policy adds support voxels while preserving the original shape as much as possible.
6. A greedy layer-wise mapper converts voxels into standard brick parts.
7. Surface finishing swaps exposed regions into tiles, slopes, grilles, or round bricks where supported by the brick library.
8. A support graph, cluster DAG, and intra-cluster ordering are exported to an LDraw MPD file.

The main implementation entry point is `core_pipeline.py`.

## 2. Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI + Gradio demo and `/api/generate` endpoint. |
| `core_pipeline.py` | End-to-end pipeline orchestration. |
| `modules/hunyuan_infer.py` | Thin wrapper around Hunyuan3D-2.1 geometry inference. |
| `modules/mesh_utils.py` | Mesh normalization, voxelization, solid fill, and GLB preview export. |
| `modules/brick_mapper.py` | Voxel-to-brick mapping, style finishing, and BOM generation. |
| `modules/risk_analysis.py` | Rule risk masks and high-risk brick detection. |
| `modules/high_risk_predictor.py` | Graph/point high-risk predictor used by repair. |
| `modules/rl_repair.py` | Runtime wrapper for the PPO repair checkpoint. |
| `modules/run_plan.py` | Support graph, clustering, assembly ordering, and MPD export. |
| `ppo_repair/` | PPO environment, training, inference, and LDR-to-voxel utilities. |
| `scripts/` | Training, dataset, validation, baseline, and rendering utilities. |
| `weights/` | Included model checkpoints used by the demo. |
| `image_inputs/` | Small image examples for reviewer smoke tests. |

## 3. Environment setup

Recommended GPU environment:

- Python 3.10.
- NVIDIA GPU with CUDA-capable PyTorch.
- Enough VRAM for Hunyuan3D-2.1 shape inference; 16 GB or more is recommended for comfortable review runs.

Exact environment exported from the development machine:

```bash
conda env create -f environments.yaml
conda activate hy3d2
```

Portable fallback:

```bash
pip install torch torchvision --index-url <your-pytorch-cuda-index>
pip install -r requirements.txt
```

The Hunyuan3D base model is not stored in this repository. It is loaded by `hy3dgen` from `tencent/Hunyuan3D-2.1` on first use. If the review server has no internet access, pre-download that model into the local model cache before running the demo.

## 4. Quick interactive run

From the repository root:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Suggested review case:

1. Upload `image_inputs/chair/chair_0001.jpg`.
2. Choose voxel resolution `16`.
3. Enable `PPO Repair`.
4. Click `Generate Assembly`.

The run writes:

```text
output/<task_id>/raw_mesh.glb
output/<task_id>/assembly_preview.glb
output/<task_id>/final_model.mpd
```

## 5. API run

After `python app.py` is running, the REST endpoint can be called with `curl`:

```bash
curl -F "file=@image_inputs/chair/chair_0001.jpg" "http://127.0.0.1:8000/api/generate?resolution=16&repair=true"
```

The JSON response includes:

- `mpd_file`: output MPD path.
- `assembly_mesh`: output GLB preview path.
- `brick_count`: number of exported bricks.
- `bom`: bill of materials.
- `logs`: pipeline log messages.

## 6. Direct Python run

```python
from core_pipeline import Img2BuildPipeline

pipeline = Img2BuildPipeline(device="cuda")
mpd_path, preview_path, brick_count, bom, logs = pipeline.run(
    "image_inputs/chair/chair_0001.jpg",
    task_id="review_demo",
    use_repair=True,
    resolution=16,
)
print("MPD:", mpd_path)
print("Preview:", preview_path)
print("Bricks:", brick_count)
print(bom.head())
```

## 7. Lightweight code checks

These commands do not run the heavy Hunyuan3D model, but they validate that the review checkout contains syntactically valid source files:

```bash
python -m py_compile app.py core_pipeline.py modules/brick_mapper.py modules/mesh_utils.py modules/risk_analysis.py modules/run_plan.py
python scripts/verify_lego_style_dataset.py --help
python scripts/train_hunyuan3d_lego_lora.py --help
python scripts/train_high_risk_proxy_styled.py --help
```

The PPO scripts import `sb3-contrib`, `stable-baselines3`, and `gymnasium`; install the full requirements before invoking them.

## 8. Repair model usage

The runtime pipeline automatically looks for:

```text
weights/ppo_lego_repair_final.zip
weights/high_risk_predictor_styled_best.pt
```

If the PPO checkpoint is absent, `core_pipeline.py` still runs but skips PPO repair. If the high-risk checkpoint is absent, rule-based risk analysis still runs and the learned risk hints are skipped.

Standalone PPO inference over prepared voxel samples:

```bash
python ppo_repair/inference.py \
  --model_path weights/ppo_lego_repair_final.zip \
  --dataset_root ppo_repair/data/train_voxels_16 \
  --risk_checkpoint weights/high_risk_predictor_styled_best.pt \
  --grid_size 16 \
  --episodes 3 \
  --output_dir ppo_repair/inference_outputs \
  --no_clip_reward
```

Use `--no_clip_reward` for faster smoke tests that do not need CLIP semantic scoring.

## 9. Training and reproduction commands

### 9.1 Build a LEGO-style finetuning dataset

Provide an LDraw source directory with `.ldr` or `.mpd` files. Optional image roots can be supplied for paired image conditions.

```bash
python scripts/build_lego_style_dataset.py \
  --ldr_root data/source_ldr \
  --image_roots image_inputs data/source_images \
  --output_root output/lego_style_dataset_v1 \
  --grid_size 16 \
  --train_ratio 0.9

python scripts/verify_lego_style_dataset.py \
  --dataset_root output/lego_style_dataset_v1 \
  --check_samples 8
```

### 9.2 LoRA finetune Hunyuan3D shape denoiser

```bash
python scripts/train_hunyuan3d_lego_lora.py \
  --dataset_root output/lego_style_dataset_v1 \
  --hunyuan_repo Hunyuan3D-2 \
  --model_path tencent/Hunyuan3D-2.1 \
  --subfolder hunyuan3d-dit-v2-1 \
  --output_dir runs/lego_lora_review \
  --device cuda \
  --dtype fp16 \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --max_train_samples 16 \
  --max_val_samples 4
```

The command above is a smoke configuration. Full training should remove the sample limits and increase the number of epochs.

### 9.3 Train the high-risk proxy

```bash
python scripts/train_high_risk_proxy_styled.py \
  --stablelego_root data/stablelego_50k/processed_contactpoint_refined \
  --lego_style_root output/lego_style_dataset_v1 \
  --base_checkpoint weights/high_risk_predictor_styled_best.pt \
  --output_dir output/styled_high_risk_proxy_review \
  --epochs 5 \
  --device cuda
```

### 9.4 Train PPO repair

```bash
python ppo_repair/train.py \
  --dataset_root ppo_repair/data/train_voxels_16 \
  --risk_checkpoint weights/high_risk_predictor_styled_best.pt \
  --log_dir logs/ppo_review \
  --total_timesteps 50000
```

For a very quick dataset conversion check, add `--prepare_only` and provide `--source_ldr_root data/source_ldr`.

## 10. Benchmark and comparison notes

The paper-facing comparison uses two tracks:

- Image-conditioned generation: compare input images through mesh-generating baselines and then use the same downstream brickization protocol.
- Text-conditioned generation: use strict canonical prompts for every external text baseline, without per-baseline prompt tuning.

For the text track, the intended runnable external baseline shortlist is `TRELLIS-text-large` and `Cube 3D v0.5`. The prompt file `docs/text_comparison_prompts_v1.csv` stores the canonical prompts used during qualitative comparison design.

The aggregate metrics table is stored as:

```text
all_case_metrics.csv
```

## 11. Expected limitations during review

- The complete end-to-end demo is GPU-heavy because it runs Hunyuan3D-2.1.
- The public demo caps voxel resolution at `16` to keep MPD files small and buildable.
- Text input is represented in the method and benchmark protocol; the current interactive app exposes the image-conditioned route.
- Full baseline reproduction requires installing each external baseline in its own environment.
- Generated figures, logs, and qualitative packs are not required for running the algorithm and are excluded from the anonymous code repository by default.

## 12. Preparing the anonymous repository

Use the root `.gitignore` and optional release helper:

```bash
python scripts/prepare_anonymous_release.py --output anonymous_release --dry-run
python scripts/prepare_anonymous_release.py --output anonymous_release
```

Before uploading, check:

```bash
git init
git status --short
git status --ignored --short
```

Do not commit local logs, manuscript build products, generated qualitative packs, nested `.git` folders, or local absolute-path bundles.
