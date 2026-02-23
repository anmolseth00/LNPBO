from __future__ import annotations
import joblib
import pickle


def save_surrogate(
    path: str,
    gp_model,
    scaler,
    columns,
    metadata: dict,
):
    """
    Save GP surrogate and preprocessing artifacts.
    """
    payload = {
        "gp_model": gp_model,
        "scaler": scaler,
        "columns": columns,
        "metadata": metadata,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_surrogate(path: str):
    """
    Load GP surrogate and preprocessing artifacts.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return (
        payload["gp_model"],
        payload["scaler"],
        payload["columns"],
        payload["metadata"],
    )
