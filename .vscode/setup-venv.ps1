# Provisions the project virtual environment (venv/) and installs requirements.txt.
#
# Runs automatically when the folder is opened in VS Code (see .vscode/tasks.json),
# and is safe to run by hand at any time:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .vscode/setup-venv.ps1
#
# - Creates venv/ only if it is missing.
# - Reinstalls dependencies only when requirements.txt has changed since last run.

$ErrorActionPreference = 'Stop'

$root         = Split-Path -Parent $PSScriptRoot
$venv         = Join-Path $root 'venv'
$python       = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $root 'requirements.txt'
$stamp        = Join-Path $venv '.requirements.sha256'

function New-Venv {
    param([string]$Target)
    if (Get-Command py     -ErrorAction SilentlyContinue) { & py -3 -m venv $Target; return }
    if (Get-Command python -ErrorAction SilentlyContinue) { & python  -m venv $Target; return }
    if (Get-Command python3 -ErrorAction SilentlyContinue) { & python3 -m venv $Target; return }
    throw 'No Python interpreter found on PATH. Install Python 3 and reopen the folder.'
}

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment in $venv ..."
    New-Venv $venv
    if (-not (Test-Path $python)) { throw 'Virtual environment creation failed.' }
}

if (-not (Test-Path $requirements)) {
    Write-Host "No requirements.txt found at $requirements; nothing to install."
    return
}

$hash     = (Get-FileHash $requirements -Algorithm SHA256).Hash.Trim()
$previous = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { '' }

if ($hash -eq $previous) {
    Write-Host 'Dependencies already up to date; skipping install.'
    return
}

Write-Host 'Installing dependencies from requirements.txt ...'
& $python -m pip install --upgrade pip
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'pip install failed.' }

Set-Content -Path $stamp -Value $hash -Encoding ascii
Write-Host 'Dependencies installed.'
