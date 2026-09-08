#!/bin/bash
# Build and upload keycycle to PyPI, then refresh the local dev install.
#
# - Uses the package's own venv python when present (keycycle/.venv), so the
#   script works without an activated environment; falls back to python3.
# - Copies the root README next to pyproject.toml before building: the wheel
#   and sdist carry the long description without the README living in two
#   places (the copy is gitignored).
# - `twine upload --skip-existing` makes a re-run after a partial upload a
#   no-op instead of a 400.
set -e

echo "--- Starting Publish Process ---"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$ROOT_DIR/keycycle"

PYTHON="$PKG_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3 || command -v python)"
fi
echo "--- Using $PYTHON ---"

# Load PYPI_TOKEN from the root .env (value never echoed).
env_file=$(find "$ROOT_DIR" -maxdepth 1 -name "*.env" | head -n 1)
if [ -n "$env_file" ]; then
    echo "--- Loading config from $(basename "$env_file") ---"
    token=$(grep "^PYPI_TOKEN=" "$env_file" | cut -d '=' -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//')
    if [ -n "$token" ]; then
        export TWINE_USERNAME="__token__"
        export TWINE_PASSWORD="$token"
        echo "--- PYPI_TOKEN loaded ---"
    fi
fi

cd "$PKG_DIR"

# 0. Required tools
if ! "$PYTHON" -m build --version &> /dev/null; then
    echo "Error: 'build' is not installed for $PYTHON. Run: uv pip install --python $PYTHON build twine"
    exit 1
fi
if ! "$PYTHON" -m twine --version &> /dev/null; then
    echo "Error: 'twine' is not installed for $PYTHON. Run: uv pip install --python $PYTHON build twine"
    exit 1
fi

# 1. Clean previous builds
echo "--- Cleaning old artifacts ---"
rm -rf dist/ build/ *.egg-info

# 2. README for the package metadata (pyproject: readme = "README.md")
cp "$ROOT_DIR/README.md" "$PKG_DIR/README.md"

# 3. Build
echo "--- Building package ---"
"$PYTHON" -m build

# 4. Upload
echo "--- Uploading to PyPI ---"
if [ -d "dist" ]; then
    "$PYTHON" -m twine upload --non-interactive --skip-existing dist/*
else
    echo "Error: Build failed, dist directory not found."
    exit 1
fi

# 5. Refresh the local dev install as an editable checkout (non-fatal): the
#    venv then imports this source tree, never a stale copy from PyPI.
cd "$ROOT_DIR"
echo "--- Refreshing local installation (editable) ---"
if command -v uv &> /dev/null; then
    uv pip install --python "$PYTHON" -e "$PKG_DIR" || echo "(editable install failed; not fatal)"
else
    "$PYTHON" -m pip install -e "$PKG_DIR" || echo "(editable install failed; not fatal)"
fi
echo "--- Done! ---"
