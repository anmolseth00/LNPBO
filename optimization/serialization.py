"""Checkpoint save/load for all surrogate families.

Checkpoint format: a directory containing
  - ``meta.json``: version, surrogate_type, feature_columns, round_number, timestamp
  - ``model.pt`` or ``model.joblib``: serialized model
  - ``scaler.joblib``: optional fitted scaler
"""

import json
import time
from pathlib import Path

import joblib

__all__ = ["save_checkpoint", "load_checkpoint"]

# Families grouped by serialization backend
_TORCH_FAMILIES = {
    "gp", "gp_mixed", "robust_gp", "multitask_gp", "casmopolitan",
    "deep_ensemble", "sngp", "laplace", "bradley_terry", "groupdro", "vrex",
    "gp_ucb",
}
_JOBLIB_FAMILIES = {
    "xgb", "xgb_ucb", "xgb_cqr", "rf_ucb", "rf_ts", "ngboost",
    "ridge", "gp_sklearn",
}


def save_checkpoint(
    checkpoint_dir,
    model,
    surrogate_type,
    feature_columns,
    round_number=0,
    scaler=None,
    extra_metadata=None,
):
    """Save a surrogate model checkpoint to *checkpoint_dir*.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory to write checkpoint files to. Created if it doesn't exist.
    model : object
        Fitted surrogate model.
    surrogate_type : str
        Surrogate family identifier (e.g. ``"gp"``, ``"xgb_ucb"``).
    feature_columns : list[str]
        Ordered feature column names.
    round_number : int
        Current BO round number.
    scaler : object, optional
        Fitted scaler (e.g. StandardScaler).
    extra_metadata : dict, optional
        Additional key-value pairs to store in ``meta.json``.
    """
    ckpt = Path(checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": 1,
        "surrogate_type": surrogate_type,
        "feature_columns": list(feature_columns),
        "round_number": round_number,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_metadata:
        meta.update(extra_metadata)

    (ckpt / "meta.json").write_text(json.dumps(meta, indent=2))

    if scaler is not None:
        joblib.dump(scaler, ckpt / "scaler.joblib")

    if surrogate_type in _TORCH_FAMILIES:
        _save_torch(model, ckpt / "model.pt", surrogate_type)
    else:
        joblib.dump(model, ckpt / "model.joblib")


def load_checkpoint(checkpoint_dir):
    """Load a surrogate model checkpoint.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing checkpoint files.

    Returns
    -------
    model : object
        Loaded surrogate model.
    meta : dict
        Metadata from ``meta.json``.
    scaler : object or None
        Loaded scaler, or None if not saved.
    """
    ckpt = Path(checkpoint_dir)
    meta = json.loads((ckpt / "meta.json").read_text())
    surrogate_type = meta["surrogate_type"]

    scaler = None
    scaler_path = ckpt / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    if surrogate_type in _TORCH_FAMILIES:
        model = _load_torch(ckpt / "model.pt")
    else:
        model = joblib.load(ckpt / "model.joblib")

    return model, meta, scaler


def _save_torch(model, path, surrogate_type):
    """Save a torch-based model (state_dict or full model)."""
    import torch

    if hasattr(model, "state_dict"):
        torch.save(
            {"state_dict": model.state_dict(), "surrogate_type": surrogate_type},
            path,
        )
    else:
        torch.save(model, path)


def _load_torch(path):
    """Load a torch checkpoint (returns raw dict or model object)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)
