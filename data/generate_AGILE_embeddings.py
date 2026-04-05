#!/usr/bin/env python3
"""Generate AGILE (GNN) embeddings for ionizable lipid SMILES.

Uses the pretrained AGILE model to produce 512-dimensional graph neural
network (GNN) representations for ionizable lipids. AGILE is a
self-supervised GNN pretrained on ~60k molecules and fine-tuned for
molecular property prediction.

This script must be run from the AGILE repository directory using its own
virtualenv, as AGILE depends on specific PyTorch Geometric versions
incompatible with the main LNPBO environment::

    cd $AGILE_ROOT  # defaults to ../AGILE relative to LNPBO repo
    .venv/bin/python -m LNPBO.data.generate_AGILE_embeddings \\
        --input /tmp/lnpbo_il_smiles_for_agile.csv \\
        --output data/agile_embeddings.npz

The output ``.npz`` file contains ``smiles`` and ``embeddings`` arrays,
which are loaded at runtime by ``data/generate_agile_loader.py``.

References:
    Xu M. et al., "AGILE platform: a deep learning powered approach to
    accelerate LNP development for mRNA delivery," Nature Communications,
    15, 6305, 2024. DOI: 10.1038/s41467-024-50619-z
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("lnpbo")

# AGILE repo must be on path — set AGILE_ROOT env var or clone as sibling
import os

AGILE_ROOT = Path(os.environ.get("AGILE_ROOT", Path(__file__).resolve().parent.parent.parent / "AGILE"))
sys.path.insert(0, str(AGILE_ROOT))


def main():
    """Extract AGILE embeddings from a CSV of SMILES and save as ``.npz``.

    Parses ``--input`` (CSV with a ``smiles`` column), ``--output`` (path
    for the ``.npz`` file), and ``--batch-size`` (default 128) from the
    command line. Loads the pretrained AGILE checkpoint, runs inference in
    eval mode on CPU, and writes ``smiles`` and ``embeddings`` arrays to
    the output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Use AGILE's own dataset class
    from dataset.dataset_test import MolTestDataset
    from models.agile_finetune import AGILE
    from torch_geometric.data import DataLoader

    logger.info("Loading dataset...")
    dataset = MolTestDataset(args.input, target="label", task="regression")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Loading AGILE model...")
    ckpt = AGILE_ROOT / "ckpt" / "pretrained_agile_60k" / "checkpoints" / "model.pth"
    model = AGILE(task="regression", num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool="mean")
    model.load_my_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    logger.info("Extracting embeddings for %d molecules...", len(dataset))
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            h, _ = model(batch)
            embeddings.append(h.cpu().numpy())
            if (i + 1) % 10 == 0:
                logger.debug("batch %d/%d", i + 1, len(loader))

    embeddings = np.concatenate(embeddings, axis=0)
    smiles_arr = np.array(dataset.smiles_data)

    logger.info("Embeddings: %s", embeddings.shape)
    np.savez_compressed(args.output, smiles=smiles_arr, embeddings=embeddings)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
