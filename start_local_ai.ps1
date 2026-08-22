[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "The local environment is missing. Run setup_local_ai.ps1 first."
}

if (-not (Test-Path "local_index\index.sqlite3")) {
    Write-Warning "The paper/book index has not been built. Data-only local questions will work, but theory modes will not."
}

& $Python -m streamlit run app.py
if ($LASTEXITCODE -ne 0) { throw "The Streamlit dashboard stopped with an error." }
