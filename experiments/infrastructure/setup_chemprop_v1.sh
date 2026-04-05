#!/bin/bash
# Create a dedicated venv for chemprop v1 (required by LiON).
# chemprop v1 requires Python <=3.10 and is incompatible with chemprop v2.
# This venv is used ONLY via subprocess from generate_LiON_fingerprints.py.

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv_chemprop_v1"

echo "=== Setting up chemprop v1 venv at $VENV_DIR ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv with Python 3.10..."
    uv venv --python 3.10 "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

echo "Installing chemprop v1 and dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
    "chemprop==1.7.1" \
    "torch>=1.9,<2.1" \
    "rdkit" \
    "scikit-learn<1.4" \
    "numpy<2" \
    "pandas<2.1"

echo ""
echo "Verifying installation..."
"$VENV_DIR/bin/python" -c "
import chemprop
print('chemprop version:', chemprop.__version__)
import torch
print('torch version:', torch.__version__)
"

echo ""
echo "=== Done ==="
echo "Venv: $VENV_DIR"
echo "chemprop_fingerprint: $VENV_DIR/bin/chemprop_fingerprint"
