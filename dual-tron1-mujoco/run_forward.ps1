param(
    [double]$WorldVx = 0.1,
    [double]$Duration = 9.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$forwardArgs = @(
    "-WorldVx", $WorldVx,
    "-Duration", $Duration
)
if ($Headless) {
    $forwardArgs += "-Headless"
}

& "$PSScriptRoot\scripts\run_forward.ps1" @forwardArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
