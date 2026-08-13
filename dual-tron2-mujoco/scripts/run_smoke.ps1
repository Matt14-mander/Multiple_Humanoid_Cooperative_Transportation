$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $project "src"
Set-Location $project
python -m pytest -q
python -m dual_tron2_mujoco.run_sim --headless --duration 1 --rebuild

