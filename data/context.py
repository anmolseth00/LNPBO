"""Experimental context: one-hot encoding, assay type inference, and
metadata columns for LNPDB data.

Context columns capture the experimental setup (cell type, target organ,
route of administration, cargo, measurement method, batching design).
These are critical covariates: the same formulation can have very different
Experiment_value depending on the assay system.
"""

import pandas as pd

# ---- Assay type classification ----

ASSAY_TYPES = [
    "in_vitro_single_formulation",
    "in_vitro_barcode_screen",
    "in_vivo_liver",
    "in_vivo_other",
    "unknown",
]


def infer_assay_type_row(row: pd.Series) -> str:
    """Classify a single row into an assay type category.

    Uses Model, Route_of_administration, Model_target, and
    Experiment_batching columns to infer whether the experiment is
    in vitro (single formulation or barcoded screen) or in vivo
    (liver-targeted or other organ).

    Args:
        row: A pandas Series with LNPDB metadata columns.

    Returns:
        One of the strings in ``ASSAY_TYPES``.
    """
    model = str(row.get("Model") or "").lower()
    route = str(row.get("Route_of_administration") or "").lower()
    target = str(row.get("Model_target") or "").lower()
    batching = str(row.get("Experiment_batching") or "").lower()

    in_vitro = model == "in_vitro" or route == "in_vitro" or target == "in_vitro"
    if in_vitro:
        if batching == "barcoded":
            return "in_vitro_barcode_screen"
        return "in_vitro_single_formulation"

    in_vivo = model == "in_vivo" or route in {
        "intravenous",
        "intramuscular",
        "intratracheal",
        "intradermal",
    }
    if in_vivo:
        if target == "liver":
            return "in_vivo_liver"
        return "in_vivo_other"

    return "unknown"


def add_assay_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``assay_type`` column to *df* using :func:`infer_assay_type_row`."""
    df = df.copy()
    df["assay_type"] = df.apply(infer_assay_type_row, axis=1)
    return df


# ---- Context columns and encoding ----

CONTEXT_COLUMNS = [
    "Model_type",
    "Model_target",
    "Route_of_administration",
    "Cargo_type",
    "Experiment_method",
    "Experiment_batching",
]


def encode_context(
    df: pd.DataFrame,
    levels: dict[str, list[str]] | None = None,
    min_count: int = 5,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """One-hot encode experimental context columns.

    Parameters
    ----------
    df : DataFrame
        Must contain some or all of CONTEXT_COLUMNS.
    levels : dict, optional
        Pre-learned levels from training data. If None, levels are learned
        from ``df`` (use this for training; pass the returned levels when
        encoding the candidate pool to ensure consistent columns).
    min_count : int
        Minimum occurrences for a level to get its own column (only used
        when learning levels from ``df``).

    Returns
    -------
    df : DataFrame
        Input dataframe with one-hot columns appended.
    context_feature_cols : list[str]
        Names of the new one-hot columns (sorted, deterministic order).
    levels : dict[str, list[str]]
        Learned or passed-through levels dict, suitable for re-use on
        the candidate pool.
    """
    if levels is None:
        levels = _learn_levels(df, min_count=min_count)

    new_cols: list[str] = []
    df = df.copy()

    for col, col_levels in sorted(levels.items()):
        if col not in df.columns:
            for lv in col_levels:
                col_name = f"ctx_{col}__{lv}"
                df[col_name] = 0
                new_cols.append(col_name)
            continue
        for lv in col_levels:
            col_name = f"ctx_{col}__{lv}"
            df[col_name] = (df[col] == lv).astype(int)
            new_cols.append(col_name)

    new_cols = sorted(new_cols)
    return df, new_cols, levels


def _learn_levels(
    df: pd.DataFrame,
    min_count: int = 5,
) -> dict[str, list[str]]:
    """Learn categorical levels from data."""
    levels: dict[str, list[str]] = {}
    for col in CONTEXT_COLUMNS:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        valid = sorted(counts[counts >= min_count].index.tolist())
        if valid:
            levels[col] = valid
    return levels
