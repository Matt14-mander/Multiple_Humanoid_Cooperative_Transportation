$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
$python = ".\.venv-sim\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $python = ".\.venv-limx38\Scripts\python.exe"
}
if (-not (Test-Path $python)) {
    throw "Run scripts\setup_windows.ps1 first."
}

$env:PYTHONNOUSERSITE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

& $python -m dual_tron1_mujoco.build_scene
if ($LASTEXITCODE -ne 0) {
    throw "Model generation failed with exit code $LASTEXITCODE."
}

& $python -c "import numpy, mujoco; print('Python/MuJoCo import OK:', mujoco.__version__, 'NumPy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "MuJoCo import failed with exit code $LASTEXITCODE."
}

$testTemp = Join-Path "runs" (
    "pytest-" + $PID + "-" + (Get-Date -Format "yyyyMMddHHmmssfff")
)
& $python -m pytest -q --cache-clear --basetemp=$testTemp
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE."
}

& $python -m dual_tron1_mujoco.run_sim --headless --duration 0.1
if ($LASTEXITCODE -ne 0) {
    throw "Headless simulation failed with exit code $LASTEXITCODE."
}
} finally {
    Pop-Location
}
