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
# setuptools<82 keeps pkg_resources available — hyperopt's atpe.py imports it
# at module load time and setuptools 82+ ships without it.
uv pip install --python "$VENV_DIR/bin/python" \
    "setuptools<82" \
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
echo "=== Pre-computing LiON cache ==="
# Populate the on-disk LiON cache once so parallel ablation workers all hit
# cache instead of each spawning chemprop_fingerprint (which would race for
# GPU memory). Skips if the LNPDB CSV isn't available — workers will then
# populate the cache lazily on first hit.
LNPDB_CSV="$REPO_ROOT/data/LNPDB_repo/data/LNPDB_for_LiON/LNPDB.csv"
if [ -f "$LNPDB_CSV" ]; then
    (cd "$REPO_ROOT" && uv run python -m data.generate_LiON_fingerprints --cache-name IL) \
        || echo "WARNING: LiON precompute failed; workers will compute lazily"
else
    echo "Skipping precompute: $LNPDB_CSV not found"
    echo "Workers will populate data/lion_cache/IL.npz lazily on first use."
fi

echo ""
echo "=== Done ==="
echo "Venv: $VENV_DIR"
echo "chemprop_fingerprint: $VENV_DIR/bin/chemprop_fingerprint"
echo "LiON cache: $REPO_ROOT/data/lion_cache/"
