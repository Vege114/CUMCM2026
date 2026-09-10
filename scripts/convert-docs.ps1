# Requires MINERU_TOKEN in the current process environment; never put it in this file.
# Writes fresh machine output to a separate ignored directory to preserve reviewed Markdown.
param([string]$OutputDirectory = '.work/mineru-refresh')
$ErrorActionPreference = 'Stop'
if (-not $env:MINERU_TOKEN) { throw 'Set MINERU_TOKEN in the process environment first.' }
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $sources = Get-ChildItem -LiteralPath 'B题' -Recurse -File |
        Where-Object { $_.Extension -in '.pdf', '.docx', '.doc', '.ppt', '.pptx', '.html' -and $_.Name -notlike '~$*' }
    foreach ($source in $sources) {
        & mineru-open-api extract $source.FullName --format md --language ch --output "$OutputDirectory/" --timeout 900
        if ($LASTEXITCODE -ne 0) { throw "Conversion failed: $($source.Name)" }
    }
} finally { Pop-Location }
