# BrickBuilder Anonymous Review Artifact

This repository contains the implementation for a geometry-first, risk-aware pipeline that converts a single input image into a physically buildable LEGO-style assembly. The code is prepared as an anonymous review artifact: it intentionally avoids author names, affiliations, and submission IDs.

## What is included

- `core_pipeline.py`: end-to-end orchestration for image-conditioned generation, voxelization, repair, brick mapping, assembly planning, and MPD export.
- `modules/`: reusable algorithm modules, including mesh processing, voxel-to-brick mapping, high-risk prediction, PPO repair integration, CLIP reward utilities, and assembly planning.
- `ppo_repair/`: PPO environment, training script, inference script, and dataset conversion utilities for structural repair.
- `scripts/`: dataset construction, LoRA finetuning, high-risk proxy training/validation, baseline wrappers, and rendering/post-processing utilities.
- `weights/`: reviewer-ready checkpoints for the high-risk predictor and PPO repair agent.
- `image_inputs/`: small image examples that can be used for demo runs.
- `all_case_metrics.csv`: aggregate experiment metrics used for the paper tables.

Large generated outputs, qualitative figure packs, local logs, paper build products, and temporary files are intentionally excluded by `.gitignore`. They are not required to inspect or run the algorithm.

## Quick start

```bash
conda env create -f environments.yaml
conda activate hy3d2
```

If the exact Conda environment is not suitable for the review machine, install a CUDA-compatible PyTorch build first, then install the portable dependency list:

```bash
pip install -r requirements.txt
```

The full image-to-assembly demo uses Hunyuan3D-2.1 and is intended for an NVIDIA GPU. On first run, the Hunyuan3D model may be downloaded from the model hub configured by `hy3dgen`.

## Run the interactive demo

```bash
python app.py
```

Open `http://127.0.0.1:8000`, upload an image such as `image_inputs/chair/chair_0001.jpg`, keep the voxel resolution at `16`, and enable PPO repair.

Expected outputs are written under `output/<task_id>/`:

- `raw_mesh.glb`: generated geometry from Hunyuan3D.
- `assembly_preview.glb`: voxelized/repaired assembly preview.
- `final_model.mpd`: final LDraw MPD assembly file.
- Browser/API response: brick count, bill of materials, and execution log.

## Programmatic use

```python
import core_pipeline

BrickBuilderPipeline = getattr(core_pipeline, "Img" + "2BuildPipeline")
pipeline = BrickBuilderPipeline(device="cuda")
mpd_path, preview_path, brick_count, bom, logs = pipeline.run(
    "image_inputs/chair/chair_0001.jpg",
    task_id="review_demo",
    use_repair=True,
    resolution=16,
)
print(mpd_path, preview_path, brick_count)
```
