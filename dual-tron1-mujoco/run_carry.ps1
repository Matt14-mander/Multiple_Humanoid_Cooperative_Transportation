param(
    [double]$WorldVx = 0.02,
    [double]$Duration = 9.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$carryArgs = @(
    "-WorldVx", $WorldVx,
    "-Duration", $Duration
)
if ($Headless) {
    $carryArgs += "-Headless"
}

& "$PSScriptRoot\scripts\run_carry.ps1" @carryArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
