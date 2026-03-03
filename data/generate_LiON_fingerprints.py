from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = MODULE_DIR / "LiON_for_LNPBO" / "trained_model_checkpoints"
DEFAULT_FEATURES_CSV = MODULE_DIR / "LiON_for_LNPBO" / "template_extra_data.csv"


def lion_fingerprints(
    smiles: list[str],
    experiment_values: list[float],
    checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    features_csv: Path | str = DEFAULT_FEATURES_CSV,
    conda_env: str = "lnp_ml",
):
    """
    Generate LiON fingerprints using Chemprop.

    :param smiles: IL SMILES
    :type smiles: list[str]
    :param experiment_values: target values
    :type experiment_values: list[float]
    :param checkpoint_dir: directory of model
    :type checkpoint_dir: str
    :param features_csv: LNP standard features
    :type features_csv: str
    :param conda_env: conda environment name
    :type conda_env: str
    """

    if len(smiles) != len(experiment_values):
        raise ValueError("SMILES and experiment_values must have same length")

    checkpoint_dir = Path(checkpoint_dir).resolve()
    features_csv = Path(features_csv).resolve()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if not features_csv.exists():
        raise FileNotFoundError(f"Features CSV not found: {features_csv}")

    def _prepare_features_csv(features_csv: Path, n_rows: int, tmpdir: Path) -> Path:
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

        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "chemprop_fingerprint",
            "--checkpoint_dir",
            str(checkpoint_dir),
            "--test_path",
            str(input_csv),
            "--features_path",
            str(features_tmp),
            "--preds_path",
            str(output_csv),
        ]

        # Conda-run Chemprop fingerprint command
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print("\n[LiON fingerprinting failed]")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

        lion_df = pd.read_csv(output_csv)

    # Drop non-feature columns if present
    feature_cols = [c for c in lion_df.columns if not c.startswith(("IL_SMILES", "Experiment_value"))]

    lion_features = lion_df[feature_cols].to_numpy()
    lion_scaler = StandardScaler()
    lion_scaled = lion_scaler.fit_transform(lion_features)

    return lion_scaled, lion_scaler
