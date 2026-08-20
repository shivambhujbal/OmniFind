# Start the FastAPI sidecar for local development.
# Production equivalent: the Tauri shell spawns the PyInstaller exe with --port.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\shiva\.venvs\fyp\Scripts\python.exe"
if ($env:FS_VENV_PYTHON) { $python = $env:FS_VENV_PYTHON }

$env:PYTHONPATH = Join-Path $root "sidecar"
Push-Location $root
try {
    & $python -m app.main --port $(if ($env:FS_PORT) { $env:FS_PORT } else { 8756 })
} finally {
    Pop-Location
}
