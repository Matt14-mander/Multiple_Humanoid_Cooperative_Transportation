# Rebuild all models with correct paths
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "=== Rebuilding Dual TRON1 Models ===" -ForegroundColor Cyan

$projectRoot = $PSScriptRoot
cd $projectRoot

# Check virtual environment
$venvPython = ".\.venv-sim\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\install.ps1" -ForegroundColor Yellow
    exit 1
}

# Check upstream projects
$workspaceRoot = Split-Path -Parent $projectRoot
$baseMjcf = Join-Path $workspaceRoot "tron1-mujoco-sim\robot-description\pointfoot\WF_TRON1A\xml\robot.xml"
$armUrdf = Join-Path $workspaceRoot "tron1-rl-deploy-arm\src\robot-description\pointfoot\WF_TRON1A\urdf\robot_with_arm.urdf"

Write-Host "`nChecking upstream projects..." -ForegroundColor Yellow
Write-Host "Looking for:" -ForegroundColor Gray
Write-Host "  - tron1-mujoco-sim: $baseMjcf" -ForegroundColor Gray
Write-Host "  - tron1-rl-deploy-arm: $armUrdf" -ForegroundColor Gray

if (-not (Test-Path $baseMjcf)) {
    Write-Host "`n✗ tron1-mujoco-sim not found in workspace root!" -ForegroundColor Red
    Write-Host "Expected location: $workspaceRoot\tron1-mujoco-sim\" -ForegroundColor Yellow
    Write-Host "`nPlease move tron1-mujoco-sim to the workspace root (same level as dual-tron1-mujoco)" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $armUrdf)) {
    Write-Host "`n✗ tron1-rl-deploy-arm not found in workspace root!" -ForegroundColor Red
    Write-Host "Expected location: $workspaceRoot\tron1-rl-deploy-arm\" -ForegroundColor Yellow
    Write-Host "`nPlease move tron1-rl-deploy-arm to the workspace root (same level as dual-tron1-mujoco)" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ All upstream projects found" -ForegroundColor Green

# Build all models
$env:PYTHONNOUSERSITE = "1"

$models = @(
    @{ name = "forward"; config = "configs\wf_dual_forward.json"; output = "models\generated\dual_tron1_forward.xml" },
    @{ name = "carry"; config = "configs\wf_dual_carry.json"; output = "models\generated\dual_tron1_carry.xml" },
    @{ name = "carry_hold"; config = "configs\wf_dual_carry_hold.json"; output = "models\generated\dual_tron1_carry_hold.xml" },
    @{ name = "carry_balance"; config = "configs\wf_dual_carry_balance.json"; output = "models\generated\dual_tron1_carry_balance.xml" }
)

Write-Host "`nBuilding models..." -ForegroundColor Yellow

foreach ($model in $models) {
    Write-Host "  Building $($model.name)..." -ForegroundColor Cyan
    & $venvPython -m dual_tron1_mujoco.build_scene --config $model.config --output $model.output

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Failed to build $($model.name)" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ $($model.output)" -ForegroundColor Green
}

Write-Host "`n✓ All models rebuilt successfully!" -ForegroundColor Green
Write-Host "`nYou can now run: .\quick_run.ps1" -ForegroundColor Cyan
