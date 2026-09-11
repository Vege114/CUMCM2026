param(
    [Parameter(Mandatory=$true)][ValidateSet(3,4)][int]$Problem,
    [Parameter(Mandatory=$true)][string]$CaseCode,
    [string]$Name = ("practice_q{0}_{1}" -f $Problem, (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$Python = "python",
    [string]$Config = "",
    [string]$BaseUrl = "http://127.0.0.1:2026",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$cliArgs = @("$PSScriptRoot/run.py", "--backend", "official", "--mode", "practice",
             "--problem", "$Problem", "--case-code", $CaseCode, "--name", $Name, "--base-url", $BaseUrl)
if ($Config) { $cliArgs += @("--config", $Config) }
if ($DryRun) { $cliArgs += "--dry-run" }
& $Python @cliArgs
exit $LASTEXITCODE
