"""Generate LiON fingerprints via the Chemprop v1 D-MPNN model.

LiON (Lipid Nanoparticle) fingerprints are penultimate-layer neural
embeddings from a directed message passing neural network (D-MPNN) trained
on ionizable lipid transfection data. The underlying architecture follows
ChemProp (Yang et al., 2019), fine-tuned on LNP-specific data by
Witten et al. (2025).

Because ChemProp v1 requires an older Python/PyTorch environment
incompatible with the main LNPBO virtualenv, fingerprint extraction is
performed via subprocess in a dedicated venv
(``experiments/infrastructure/setup_chemprop_v1.sh``).

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

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lnpbo")

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = MODULE_DIR / "LiON_for_LNPBO" / "trained_model_checkpoints"
DEFAULT_FEATURES_CSV = MODULE_DIR / "LiON_for_LNPBO" / "template_extra_data.csv"


DEFAULT_CHEMPROP_VENV = MODULE_DIR.parent / ".venv_chemprop_v1"


def lion_fingerprints(
    smiles: list[str],
    experiment_values: list[float] | None,
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    features_csv: Path | str = DEFAULT_FEATURES_CSV,
    chemprop_venv: Path | str | None = None,
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

    if experiment_values is None:
        experiment_values = [0.0] * len(smiles)

    if len(smiles) != len(experiment_values):
        raise ValueError("SMILES and experiment_values must have same length")

    checkpoint_dir = Path(checkpoint_dir).resolve()
    features_csv = Path(features_csv).resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    def _prepare_features_csv(features_csv: Path, n_rows: int, tmpdir: Path) -> Path:
        """Prepare the auxiliary features CSV for ChemProp input.

        The template CSV may contain a single row (broadcast to all
        molecules) or exactly ``n_rows`` rows. The result is written to
        a temporary file in ``tmpdir``.

        Args:
            features_csv: Path to the template features CSV.
            n_rows: Expected number of rows (one per SMILES).
            tmpdir: Temporary directory for the output file.

        Returns:
            Path to the prepared features CSV.

        Raises:
            ValueError: If the template has neither 1 nor ``n_rows`` rows.
        """
        features_df = pd.read_csv(features_csv)

        if len(features_df) == n_rows:
            pass
        elif len(features_df) == 1:
            features_df = pd.concat([features_df] * n_rows, ignore_index=True)
        else:
            raise ValueError(f"features.csv has {len(features_df)} rows, expected {n_rows}")

        out = tmpdir / "lion_features.csv"
        features_df.to_csv(out, index=False)
        return out

    # Create input spreadsheet
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        input_csv = tmpdir / "lion_input.csv"
        output_csv = tmpdir / "lion_output.csv"

        df = pd.DataFrame(
            {
                "IL_SMILES": smiles,
                "Experiment_value": experiment_values,
            }
        )
        df.to_csv(input_csv, index=False)

        features_tmp = _prepare_features_csv(
            features_csv,
            n_rows=len(smiles),
            tmpdir=tmpdir,
        )

        venv = Path(chemprop_venv).resolve() if chemprop_venv else DEFAULT_CHEMPROP_VENV
        chemprop_bin = venv / "bin" / "chemprop_fingerprint"
        if not chemprop_bin.exists():
            raise FileNotFoundError(
                f"chemprop_fingerprint not found at {chemprop_bin}. "
                f"Create the venv with: experiments/infrastructure/setup_chemprop_v1.sh"
            )

        cmd = [
            str(chemprop_bin),
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--test_path",
            str(input_csv),
            "--features_path",
            str(features_tmp),
            "--preds_path",
            str(output_csv),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("LiON fingerprinting failed")
            logger.warning("STDOUT:\n%s", e.stdout)
            logger.warning("STDERR:\n%s", e.stderr)
            raise

        lion_df = pd.read_csv(output_csv)

    # Drop non-feature columns if present
    feature_cols = [c for c in lion_df.columns if not c.startswith(("IL_SMILES", "Experiment_value"))]

    lion_features = lion_df[feature_cols].to_numpy()
    lion_scaler = StandardScaler()
    lion_scaled = lion_scaler.fit_transform(lion_features)

    return lion_scaled, lion_scaler
