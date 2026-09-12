param(
    [ValidateSet('tf', 'rl')]
    [string]$Profile
)
$ErrorActionPreference = 'Stop'
# Keep this a simple script: advanced parameters would consume Python's -v as -Verbose.
$commandArgs = @($args)
if (-not $Profile -or $commandArgs.Count -eq 0) {
    throw 'Usage: .\scripts\run_ml.ps1 {tf|rl} COMMAND [ARGUMENTS...]'
}
$repository = Split-Path -Parent $PSScriptRoot
# All program paths after the profile are Linux paths or paths relative to the repository.
& wsl.exe -d Ubuntu --cd $repository -- bash scripts/run_ml.sh $Profile @commandArgs
exit $LASTEXITCODE
