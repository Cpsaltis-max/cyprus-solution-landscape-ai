[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/ and select 'Add Python to PATH'."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was not found. Install it from https://ollama.com/download/windows/ and run this script again."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating the private Python environment..."
    & py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python could not create the virtual environment." }
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "Installing the local dashboard requirements..."
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip could not be upgraded." }
& $Python -m pip install -r requirements-local.txt
if ($LASTEXITCODE -ne 0) { throw "The Python requirements could not be installed." }

New-Item -ItemType Directory -Force -Path "private_corpus\papers" | Out-Null
New-Item -ItemType Directory -Force -Path "private_corpus\book" | Out-Null
New-Item -ItemType Directory -Force -Path "local_index" | Out-Null

Write-Host "Downloading the initial local answer model..."
& ollama pull qwen3:8b
if ($LASTEXITCODE -ne 0) { throw "Ollama could not download qwen3:8b." }

Write-Host "Downloading the multilingual embedding model..."
& ollama pull bge-m3
if ($LASTEXITCODE -ne 0) { throw "Ollama could not download bge-m3." }

Write-Host ""
Write-Host "Setup completed successfully." -ForegroundColor Green
Write-Host "Next: copy the 45 papers into private_corpus\papers and the book into private_corpus\book."
Write-Host "Then run:  .\.venv\Scripts\python.exe build_local_index.py"
Write-Host "Finally run: .\start_local_ai.ps1"
