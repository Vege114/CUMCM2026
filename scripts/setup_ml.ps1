param(
    [ValidateSet('all', 'tf', 'rl')]
    [string]$Profile = 'all',
    [string]$Distribution = 'Ubuntu'
)
$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
& wsl.exe -d $Distribution --cd $repository -- bash scripts/setup_ml.sh $Profile
exit $LASTEXITCODE
