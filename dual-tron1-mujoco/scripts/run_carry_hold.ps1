param(
    [double]$PayloadMass = 0.5,
    [double]$Duration = 10.0,
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
    if ($PayloadMass -le 0.0) {
        throw "PayloadMass must be positive."
    }
    $env:PYTHONNOUSERSITE = "1"
    $simArgs = @(
        "-m", "dual_tron1_mujoco.run_sim",
        "--carry-hold-test",
        "--payload-mass", $PayloadMass,
        "--duration", $Duration
    )
    if ($Headless) {
        $simArgs += "--headless"
    }
    & $python @simArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Carry-hold simulation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
