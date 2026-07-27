$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "Python launcher 'py' was not found. Install CPython 3.8 x64 first."
    }

    py -3.8 -m venv .venv-limx38
    $python = ".\.venv-limx38\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements-win.txt
    & $python -m pip install -e ".[policy,test]"
    & $python -m pip install `
        ..\tron1-mujoco-sim\limxsdk-lowlevel\python3\win\limxsdk-3.4.0-py3-none-any.whl `
        --no-deps

    Write-Host "LIMX SDK-compatible environment ready: $projectRoot\.venv-limx38"
} finally {
    Pop-Location
}
