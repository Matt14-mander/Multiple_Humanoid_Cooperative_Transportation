# Install dual-tron1-mujoco project
$ErrorActionPreference = "Stop"

Write-Host "=== Installing dual-tron1-mujoco Project ===" -ForegroundColor Cyan

$projectRoot = "E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\dual-tron1-mujoco"
cd $projectRoot

# Check virtual environment
$venvPython = ".\.venv-sim\Scripts\python.exe"
$venvPip = ".\.venv-sim\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Error: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: .\scripts\setup_windows.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Virtual environment found: Python $(&$venvPython --version)" -ForegroundColor Green

# Install project
Write-Host "`nInstalling project..." -ForegroundColor Yellow
& $venvPip install -e ".[policy]"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Installation complete!" -ForegroundColor Green

    # Verify installation
    Write-Host "`nVerifying installation..." -ForegroundColor Yellow
    & $venvPython -c "import dual_tron1_mujoco; print('✓ Module imported successfully')"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nYou can now run the simulation!" -ForegroundColor Green
        Write-Host "Command: .\quick_run.ps1" -ForegroundColor Cyan
    }
} else {
    Write-Host "`n✗ Installation failed" -ForegroundColor Red
    exit 1
}
