#!/usr/bin/env bash
set -e

# Determine python binary
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    echo "WARNING: python3.12 not found. Falling back to $(python3 --version)."
    PYTHON_BIN="python3"
else
    echo "ERROR: Neither python3.12 nor python3 found."
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

# Create .venv
echo "Creating virtual environment in .venv..."
$PYTHON_BIN -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA (cu126 -> cu124 -> cu121)
echo "Installing PyTorch with CUDA..."
if pip install torch --index-url https://download.pytorch.org/whl/cu126; then
    echo "Successfully installed PyTorch with cu126."
elif pip install torch --index-url https://download.pytorch.org/whl/cu124; then
    echo "Successfully installed PyTorch with cu124."
elif pip install torch --index-url https://download.pytorch.org/whl/cu121; then
    echo "Successfully installed PyTorch with cu121."
else
    echo "ERROR: Failed to install PyTorch with cu126, cu124, or cu121."
    exit 1
fi

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. Virtual environment ready in .venv."
