$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python -m tron2_chip.run_sanity --duration 1.5 --rebuild
}
finally {
    Pop-Location
}

