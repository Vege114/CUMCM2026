$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
    throw 'latexmk is missing. Install TeX Live with XeLaTeX and Chinese language support.'
}
Push-Location (Join-Path $repoRoot 'paper')
try {
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX build failed: $LASTEXITCODE" }
} finally { Pop-Location }
