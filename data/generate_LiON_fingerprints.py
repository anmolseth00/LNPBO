"""Generate LiON fingerprints via the Chemprop v1 D-MPNN model.

LiON (Lipid Nanoparticle) fingerprints are penultimate-layer neural
embeddings from a directed message passing neural network (D-MPNN) trained
on ionizable lipid transfection data. The underlying architecture follows
ChemProp (Yang et al., 2019), fine-tuned on LNP-specific data by
Witten et al. (2025).

Because ChemProp v1 requires an older Python/PyTorch environment
incompatible with the main LNPBO virtualenv, fingerprint extraction is
performed via subprocess in a dedicated ChemProp v1 virtualenv (by default
``.venv_chemprop_v1`` alongside the package; override with ``chemprop_venv``).

References:
    Yang K. et al., "Analyzing Learned Molecular Representations for
    Property Prediction," Journal of Chemical Information and Modeling,
    59(8), 3370-3388, 2019. DOI: 10.1021/acs.jcim.9b00237

    Witten J. et al., "Artificial intelligence-guided design of lipid
    nanoparticles for pulmonary gene therapy," Nature Biotechnology, 43(11),
    1790-1799, 2025. DOI: 10.1038/s41587-024-02490-y
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cache_utils import cached_encode

logger = logging.getLogger("lnpbo")

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = MODULE_DIR / "LiON_for_LNPBO" / "trained_model_checkpoints"
DEFAULT_FEATURES_CSV = MODULE_DIR / "LiON_for_LNPBO" / "template_extra_data.csv"

CACHE_DIR = MODULE_DIR / "lion_cache"

DEFAULT_CHEMPROP_VENV = MODULE_DIR.parent / ".venv_chemprop_v1"


def lion_fingerprints(
    smiles: list[str],
    experiment_values: list[float] | None = None,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    features_csv: Path | str = DEFAULT_FEATURES_CSV,
    chemprop_venv: Path | str | None = None,
    cache_name: str = "default",
    scaler: StandardScaler | None = None,
):
    """Generate LiON fingerprints by extracting D-MPNN penultimate-layer embeddings.

    Writes SMILES and experiment values to a temporary CSV, invokes the
    ``chemprop_fingerprint`` CLI in a separate virtualenv, reads back the
    output embeddings, and returns them as a StandardScaler-normalized matrix.

    Args:
        smiles: List of ionizable lipid SMILES strings.
        experiment_values: Per-SMILES target values passed to ChemProp
            (used for input formatting; the model does not train on these).
            If ``None``, zeros are substituted.
        checkpoint_dir: Path to the directory containing LiON model
            checkpoint files (``*.pt``).
        features_csv: Path to the template extra features CSV required
            by the LiON model (contains auxiliary descriptors).
        chemprop_venv: Path to the ChemProp v1 virtualenv. Defaults to
            ``{project_root}/.venv_chemprop_v1``.

    Returns:
        Tuple of ``(lion_scaled, lion_scaler)`` where ``lion_scaled`` is an
        ndarray of shape ``(n_molecules, embedding_dim)`` with z-scored
        embeddings, and ``lion_scaler`` is the fitted ``StandardScaler``.

    Raises:
        ValueError: If ``smiles`` and ``experiment_values`` have different
            lengths.
        FileNotFoundError: If checkpoint directory, features CSV, or
            ChemProp binary is not found.
        subprocess.CalledProcessError: If the ChemProp subprocess fails.

    References:
        Yang K. et al., JCIM 59(8), 3370-3388, 2019 (D-MPNN architecture).

        Witten J. et al., Nature Biotechnology 43(11), 1790-1799, 2025
        (LiON model and training data).
    """

    # experiment_values are only used to format the chemprop input CSV;
    # the fingerprint forward pass does not depend on them. We can therefore
    # cache by SMILES alone and ignore whatever values the caller passes.
    del experiment_values

    checkpoint_dir = Path(checkpoint_dir).resolve()
    features_csv = Path(features_csv).resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    venv = Path(chemprop_venv).resolve() if chemprop_venv else DEFAULT_CHEMPROP_VENV
    chemprop_bin = venv / "bin" / "chemprop_fingerprint"

    def _compute(todo_smiles: list[str]) -> dict[str, np.ndarray]:
        """Run chemprop_fingerprint on cache misses and return SMILES → embedding."""
        if not chemprop_bin.exists():
            raise FileNotFoundError(
                f"chemprop_fingerprint not found at {chemprop_bin}. "
                f"Install ChemProp v1 in a dedicated virtualenv (default: "
                f"{DEFAULT_CHEMPROP_VENV}) and point chemprop_venv at it."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_csv = tmpdir / "lion_input.csv"
            output_csv = tmpdir / "lion_output.csv"

            pd.DataFrame(
                {
                    "IL_SMILES": todo_smiles,
                    # Zeros: chemprop wants the column but ignores values
                    # during fingerprint extraction.
                    "Experiment_value": [0.0] * len(todo_smiles),
                }
            ).to_csv(input_csv, index=False)

            features_df = pd.read_csv(features_csv)
            if len(features_df) == 1:
                features_df = pd.concat([features_df] * len(todo_smiles), ignore_index=True)
            elif len(features_df) != len(todo_smiles):
                raise ValueError(
                    f"features.csv has {len(features_df)} rows, expected 1 or {len(todo_smiles)}"
                )
            features_tmp = tmpdir / "lion_features.csv"
            features_df.to_csv(features_tmp, index=False)

            cmd = [
                str(chemprop_bin),
                "--checkpoint_dir", str(checkpoint_dir),
                "--test_path", str(input_csv),
                "--features_path", str(features_tmp),
                "--preds_path", str(output_csv),
                # Pin to CPU: fingerprinting is a small forward pass and the
                # parallel ablation workers may already be holding GPU memory.
                "--no_cuda",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.warning("LiON fingerprinting failed")
                logger.warning("STDOUT:\n%s", e.stdout)
                logger.warning("STDERR:\n%s", e.stderr)
                raise

            lion_df = pd.read_csv(output_csv)

        feature_cols = [
            c for c in lion_df.columns
            if not c.startswith(("IL_SMILES", "Experiment_value"))
        ]
        embeddings = lion_df[feature_cols].to_numpy()
        return {s: embeddings[i] for i, s in enumerate(todo_smiles)}

    return cached_encode(
        smiles, _compute, CACHE_DIR, cache_name=cache_name,
        scaler=scaler, label="LiON embeddings",
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = argparse.ArgumentParser(description="Pre-compute LiON embeddings for LNPDB ionizable lipids")
    parser.add_argument(
        "--data-path",
        default=str(MODULE_DIR / "LNPDB_repo" / "data" / "LNPDB_for_LiON" / "LNPDB.csv"),
    )
    parser.add_argument("--cache-name", default="IL")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path, low_memory=False)
    smiles = df["IL_SMILES"].dropna()
    smiles = smiles[~smiles.isin(["None", "Unknown", ""])]
    unique = smiles.unique().tolist()
    logger.info("Pre-computing LiON embeddings for %d unique ionizable lipids", len(unique))
    lion_fingerprints(unique, cache_name=args.cache_name)
    logger.info("Cache written to %s", CACHE_DIR / f"{args.cache_name}.npz")
