# Start the React + Vite dev frontend.
# Replacement for the Streamlit frontend (run_frontend.ps1).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "frontend"

Push-Location $frontend
try {
    npm run dev
} finally {
    Pop-Location
}
