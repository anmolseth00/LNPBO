#!/usr/bin/env bash
# Setup script for COMET baseline environment.
#
# Usage:
#   ./benchmarks/baselines/setup_comet.sh /path/to/COMET
#
# This script:
#   1. Creates a Python 3.10 venv in the COMET repo
#   2. Installs all dependencies (CPU-only PyTorch for macOS)
#   3. Downloads pretrained weights from Google Drive
#   4. Verifies the installation
#
# After running this script, use the COMET baseline via:
#   python -m benchmarks.baselines.comet_wrapper --export
#   python -m benchmarks.baselines.comet_wrapper --run \
#       --comet-repo /path/to/COMET \
#       --weights-dir /path/to/COMET/experiments/weights/<model_dir>
#   python -m benchmarks.baselines.comet_wrapper --import-results

set -euo pipefail

COMET_REPO="${1:?Usage: $0 /path/to/COMET}"
COMET_REPO=$(cd "$COMET_REPO" && pwd)

echo "========================================"
echo "COMET Baseline Setup"
echo "========================================"
echo "COMET repo: $COMET_REPO"

# Step 1: Create venv
echo ""
echo "--- Step 1: Creating Python 3.10 venv ---"
cd "$COMET_REPO"
if [ -d ".venv" ]; then
    echo "  .venv already exists, skipping creation"
else
    uv venv .venv --python 3.10
fi

# Step 2: Install dependencies
echo ""
echo "--- Step 2: Installing dependencies ---"
source .venv/bin/activate

uv pip install torch torchvision torchaudio
uv pip install \
    lmdb \
    ml-collections \
    numpy==1.24.4 \
    scipy \
    tensorboardX \
    tqdm \
    tokenizers \
    pyprojroot \
    pandas \
    scikit-learn \
    rdkit-pypi \
    gdown

# Step 3: Download pretrained weights
echo ""
echo "--- Step 3: Downloading pretrained weights ---"
mkdir -p experiments/weights
mkdir -p ckp

# COMET pretrained weights (lipid LNP model)
if find experiments/weights -name "checkpoint_best.pt" 2>/dev/null | head -1 | grep -q .; then
    echo "  Pretrained weights already found, skipping download"
else
    echo "  Downloading from Google Drive..."
    echo "  URL: https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv"
    python -m gdown --folder "https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv" -O experiments/weights/ || {
        echo ""
        echo "  WARNING: Automatic download failed. Please download manually:"
        echo "  1. Go to https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv"
        echo "  2. Download the weight folders to experiments/weights/"
        echo "  3. Each folder should contain checkpoint_best.pt"
    }
fi

# Uni-Mol base checkpoint (for fine-tuning only)
if [ -f ckp/mol_pre_no_h_220816.pt ]; then
    echo "  Uni-Mol base checkpoint already found"
else
    echo "  Downloading Uni-Mol base checkpoint..."
    echo "  URL: https://drive.google.com/drive/folders/1Ul89o6Vj93T01foKa1-898H32bJLTJkm"
    python -m gdown --folder "https://drive.google.com/drive/folders/1Ul89o6Vj93T01foKa1-898H32bJLTJkm" -O ckp/ || {
        echo ""
        echo "  WARNING: Automatic download failed. This is only needed for fine-tuning."
        echo "  Zero-shot inference works without this checkpoint."
    }
fi

# Step 4: Verify installation
echo ""
echo "--- Step 4: Verifying installation ---"
python -c "
import sys
sys.path.insert(0, '.')
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import unimol
print('  COMET/unimol imported successfully')
from unimol.models.unimol import NPUniMolModel
print('  NPUniMolModel class loaded')
" 2>&1 | grep -v "not installed corrected" | grep -v "fused_"

# Check for weights
N_WEIGHTS=$(find experiments/weights -name "checkpoint_best.pt" 2>/dev/null | wc -l | tr -d ' ')
WEIGHTS_DIR=$(find experiments/weights -name "checkpoint_best.pt" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -n "$WEIGHTS_DIR" ]; then
    echo "  Weights found: $N_WEIGHTS checkpoint(s)"
    echo "  First: $WEIGHTS_DIR"
    echo ""
    echo "========================================"
    echo "Setup complete. Run baselines with:"
    echo "========================================"
    echo "  cd $(dirname "$COMET_REPO")/LNPBO"
    echo "  python -m benchmarks.baselines.comet_wrapper --export"
    echo ""
    echo "  # Single fold (uses first checkpoint found):"
    echo "  python -m benchmarks.baselines.comet_wrapper --run \\"
    echo "      --comet-repo $COMET_REPO \\"
    echo "      --weights-dir $WEIGHTS_DIR"
    echo ""
    echo "  # Or pass the parent dir to auto-select first checkpoint:"
    echo "  python -m benchmarks.baselines.comet_wrapper --run \\"
    echo "      --comet-repo $COMET_REPO \\"
    echo "      --weights-dir $COMET_REPO/experiments/weights/weights"
    echo ""
    echo "  python -m benchmarks.baselines.comet_wrapper --import-results"
else
    echo ""
    echo "========================================"
    echo "Setup partially complete (no weights found)."
    echo "========================================"
    echo "Download weights from:"
    echo "  https://drive.google.com/drive/folders/1IBz8iWrPX5Xnlb02VaTNR-7xuKuYUHZv"
    echo "Place them in: $COMET_REPO/experiments/weights/"
fi
