param(
    [double]$PayloadMass = 0.5,
    [double]$Duration = 10.0,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$balanceArgs = @(
    "-PayloadMass", $PayloadMass,
    "-Duration", $Duration
)
if ($Headless) {
    $balanceArgs += "-Headless"
}

& "$PSScriptRoot\scripts\run_carry_balance.ps1" @balanceArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
