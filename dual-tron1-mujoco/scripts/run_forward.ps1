param(
    [double]$WorldVx = 0.1,
    [double]$Duration = 9.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $python = ".\.venv-sim\Scripts\python.exe"
    if (-not (Test-Path $python)) {
        throw "Run scripts\setup_windows.ps1 first."
    }
    $env:PYTHONNOUSERSITE = "1"
    $simArgs = @(
        "-m", "dual_tron1_mujoco.run_sim",
        "--forward-test",
        "--world-vx", $WorldVx,
        "--duration", $Duration
    )
    if ($Headless) {
        $simArgs += "--headless"
    }
    & $python @simArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Forward simulation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
