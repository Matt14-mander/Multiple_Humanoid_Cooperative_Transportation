# Quick run script - Use pre-generated models (no rebuild)
param(
    [string]$Mode = "forward",  # forward/carry/hold/balance
    [switch]$Headless,
    [double]$Duration = 9.0,
    [double]$WorldVx = 0.1
)

$ErrorActionPreference = "Stop"

Write-Host "=== Dual TRON1 MuJoCo Simulation ===" -ForegroundColor Cyan
Write-Host "Current directory: $PSScriptRoot" -ForegroundColor Yellow

# Check virtual environment
$venvPython = "$PSScriptRoot\.venv-sim\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\install.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment found" -ForegroundColor Green

# Map mode to model file and config
$modelMap = @{
    "forward" = @{
        "model" = "models\generated\dual_tron1_forward.xml"
        "config" = "configs\wf_dual_forward.json"
        "controller" = "policy"
        "description" = "Forward motion test (robots will move)"
    }
    "carry" = @{
        "model" = "models\generated\dual_tron1_carry.xml"
        "config" = "configs\wf_dual_carry.json"
        "controller" = "policy"
        "description" = "Cooperative carry test"
    }
    "hold" = @{
        "model" = "models\generated\dual_tron1_carry_hold.xml"
        "config" = "configs\wf_dual_carry_hold.json"
        "controller" = "carry_hold"
        "description" = "Fixed-base payload hold test"
    }
    "balance" = @{
        "model" = "models\generated\dual_tron1_carry_balance.xml"
        "config" = "configs\wf_dual_carry_balance.json"
        "controller" = "carry_balance"
        "description" = "Free-base balance test"
    }
}

if (-not $modelMap.ContainsKey($Mode)) {
    Write-Host "Unknown mode: $Mode" -ForegroundColor Red
    Write-Host "Available modes: forward, carry, hold, balance" -ForegroundColor Yellow
    exit 1
}

$modeConfig = $modelMap[$Mode]
$modelPath = Join-Path $PSScriptRoot $modeConfig["model"]

# Check if model exists
if (-not (Test-Path $modelPath)) {
    Write-Host "Error: Model file not found: $modelPath" -ForegroundColor Red
    Write-Host "You need the upstream TRON1 model projects to rebuild." -ForegroundColor Yellow
    exit 1
}

Write-Host "`nRunning mode: $($modeConfig['description'])" -ForegroundColor Cyan

# Run simulation
Push-Location $PSScriptRoot
try {
    $env:PYTHONNOUSERSITE = "1"

    $simArgs = @(
        "-m", "dual_tron1_mujoco.run_sim",
        "--model", $modelPath,
        "--config", $modeConfig["config"],
        "--controller", $modeConfig["controller"],
        "--duration", $Duration
    )

    # Add mode-specific parameters
    if ($Mode -eq "forward" -or $Mode -eq "carry") {
        $simArgs += "--world-vx", $WorldVx
    }

    if ($Headless) {
        $simArgs += "--headless"
        Write-Host "Headless mode: No viewer window" -ForegroundColor Yellow
    } else {
        Write-Host "GUI mode: MuJoCo viewer will open" -ForegroundColor Green
    }

    Write-Host "`nStarting simulation..." -ForegroundColor Yellow
    & $venvPython @simArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Simulation completed!" -ForegroundColor Green
        Write-Host "Results saved in: runs/ directory" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ Simulation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }

} finally {
    Pop-Location
}

Write-Host "`nUsage:" -ForegroundColor Cyan
Write-Host "  .\quick_run.ps1                              # Forward motion (default)" -ForegroundColor Gray
Write-Host "  .\quick_run.ps1 -Mode carry                  # Cooperative carry" -ForegroundColor Gray
Write-Host "  .\quick_run.ps1 -Mode hold                   # Fixed-base hold" -ForegroundColor Gray
Write-Host "  .\quick_run.ps1 -Mode balance                # Free-base balance" -ForegroundColor Gray
Write-Host "  .\quick_run.ps1 -Headless                    # Headless mode" -ForegroundColor Gray
Write-Host "  .\quick_run.ps1 -Duration 15 -WorldVx 0.2    # Custom parameters" -ForegroundColor Gray
