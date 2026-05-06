$ErrorActionPreference = "Stop"

$projectRoot = "D:\tdc2\img2build_pipeline"
$logDir = Join-Path $projectRoot "ppo_repair\logs\risk_clip_16_run1000k"
$defaultWeight = Join-Path $projectRoot "weights\ppo_lego_repair_final.zip"
$trainedWeight = Join-Path $logDir "ppo_lego_repair_final.zip"
$renderOutput = Join-Path $projectRoot "output\demo10_render_v16_1000k"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $renderOutput | Out-Null

foreach ($stalePath in @(
  (Join-Path $logDir "monitor.csv"),
  (Join-Path $logDir "ppo_lego_repair_final.zip")
)) {
  if (Test-Path $stalePath) {
    Remove-Item $stalePath -Force
  }
}

Write-Host "[Run] Starting 1000k PPO training..."
conda run -n hy3d2 python (Join-Path $projectRoot "ppo_repair\train.py") `
  --dataset_root (Join-Path $projectRoot "ppo_repair\data\train_voxels_16") `
  --source_ldr_root "D:\tdc2\hunyuan3d\out_fix" `
  --risk_checkpoint "D:\tdc2\img2build2\outputs\stablelego_contactpoint_refined_full\checkpoints\best.pt" `
  --log_dir $logDir `
  --total_timesteps 1000000

if (-not (Test-Path $trainedWeight)) {
  throw "Training finished but no trained weight was found at $trainedWeight"
}

Copy-Item $trainedWeight $defaultWeight -Force
Write-Host "[Run] Copied final weight to $defaultWeight"

Write-Host "[Run] Rendering 10 demo images..."
conda run -n hy3d2 python (Join-Path $projectRoot "batch_demo_render.py") `
  --output_root $renderOutput

Write-Host "[Run] Done."
Write-Host "[Run] Log dir: $logDir"
Write-Host "[Run] Render dir: $renderOutput"
