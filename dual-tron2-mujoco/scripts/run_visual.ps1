$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $project "src"
Set-Location $project
python -m dual_tron2_mujoco.run_sim `
  --rebuild `
  --payload-mass 2 `
  --payload-com 0.05 -0.03 0.02 `
  --duration 20

