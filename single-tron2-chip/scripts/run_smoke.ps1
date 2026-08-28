$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python -m pytest -q
    python -m tron2_chip.run_sanity --headless --duration 0.25 --rebuild
    python -m tron2_chip.run_deployment --headless --duration 1.2
    python -m tron2_chip.plot_csv runs/arm_sanity_latest.csv
}
finally {
    Pop-Location
}
