param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {

    if ($PythonExe) {
        & $PythonExe -m venv .venv-sim
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv-sim
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv-sim
    } else {
        throw "Python 3.8+ x64 was not found. Install python.org CPython first."
    }

    $python = ".\.venv-sim\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    & $python -m pip install -e ".[policy,test]"

    Write-Host "ROS-free simulation environment ready: $projectRoot\.venv-sim"
} finally {
    Pop-Location
}
