$ErrorActionPreference = "Stop"

Write-Host "--- Starting Publish Process ---" -ForegroundColor Cyan

# Load PYPI_TOKEN from .env if available
$envFiles = Get-ChildItem -Path "$PSScriptRoot" -Filter "*.env" -File
if ($envFiles) {
    $envFile = $envFiles[0]
    Write-Host "--- Loading config from $($envFile.Name) ---" -ForegroundColor Cyan
    foreach ($line in Get-Content $envFile.FullName) {
        if ($line -match "^\s*PYPI_TOKEN\s*=\s*(.*)") {
            $token = $matches[1].Trim().Trim('"').Trim("'")
            $env:TWINE_USERNAME = "__token__"
            $env:TWINE_PASSWORD = $token
            Write-Host "--- PYPI_TOKEN loaded ---" -ForegroundColor Cyan
            break
        }
    }
}

# Change to the package directory
Set-Location "$PSScriptRoot/keycycle"

# 0. Check for required tools
try {
    python -m build --version | Out-Null
} catch {
    Write-Host "Error: 'build' is not installed. Run: pip install build twine" -ForegroundColor Red
    exit 1
}

# 1. Clean previous builds
Write-Host "--- Cleaning old artifacts ---" -ForegroundColor Yellow
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path build) { Remove-Item -Recurse -Force build }
Get-ChildItem -Filter "*.egg-info" | Remove-Item -Recurse -Force

# 2. Build the package
Write-Host "--- Building package ---" -ForegroundColor Yellow
python -m build

# 3. Upload to PyPI
Write-Host "--- Uploading to PyPI ---" -ForegroundColor Yellow
if (Test-Path dist) {
    python -m twine upload dist/*
} else {
    Write-Host "Error: Build failed, dist directory not found." -ForegroundColor Red
    exit 1
}
Set-Location -Path $PSScriptRoot
Write-Host "--- Refreshing local installation ---" -ForegroundColor Yellow
uv pip install --refresh -U keycycle
Write-Host "--- Done! ---" -ForegroundColor Green
