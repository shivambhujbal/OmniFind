# Start the Streamlit dev frontend (stand-in for the Tauri + React shell).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if ($env:FS_VENV_PYTHON) { $python = $env:FS_VENV_PYTHON }

& $python -m streamlit run (Join-Path $root "streamlit_app\app.py") --server.address 127.0.0.1
