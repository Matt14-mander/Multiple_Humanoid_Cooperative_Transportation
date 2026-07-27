param(
    [double]$PayloadMass = 0.5,
    [double]$Duration = 10.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$carryHoldArgs = @(
    "-PayloadMass", $PayloadMass,
    "-Duration", $Duration
)
if ($Headless) {
    $carryHoldArgs += "-Headless"
}

& "$PSScriptRoot\scripts\run_carry_hold.ps1" @carryHoldArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
