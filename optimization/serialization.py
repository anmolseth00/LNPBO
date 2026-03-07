import joblib


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
    joblib.dump(payload, path)


def load_surrogate(path: str):
    """
    Load GP surrogate and preprocessing artifacts.
    """
    payload = joblib.load(path)

    return (
        payload["gp_model"],
        payload["scaler"],
        payload["columns"],
        payload["metadata"],
    )
