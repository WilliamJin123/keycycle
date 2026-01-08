#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "--- Starting Publish Process ---"

ROOT_DIR="$(dirname "$0")"

# Check for .env file to load PYPI_TOKEN
env_file=$(find "$ROOT_DIR" -maxdepth 1 -name "*.env" | head -n 1)
if [ -n "$env_file" ]; then
    echo "--- Loading config from $(basename "$env_file") ---"
    # Extract PYPI_TOKEN, removing quotes if present
    token=$(grep "^PYPI_TOKEN=" "$env_file" | cut -d '=' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//')
    if [ -n "$token" ]; then
        export TWINE_USERNAME="__token__"
        export TWINE_PASSWORD="$token"
        echo "--- PYPI_TOKEN loaded ---"
    fi
fi

# Change to the package directory
cd "$ROOT_DIR/keycycle"

# 0. Check for required tools
if ! python -m build --version &> /dev/null; then
    echo "Error: 'build' is not installed. Run: (uv) pip install build twine"
    exit 1
fi

# 1. Clean previous builds to avoid confusion
echo "--- Cleaning old artifacts ---"
rm -rf dist/ build/ *.egg-info

# 2. Build the package
echo "--- Building package ---"
python -m build

# 3. Upload to PyPI
echo "--- Uploading to PyPI ---"
# Check if dist/* exists
if [ -d "dist" ]; then
    python -m twine upload dist/*
else
    echo "Error: Build failed, dist directory not found."
    exit 1
fi

cd..
echo "--- Done! ---"
