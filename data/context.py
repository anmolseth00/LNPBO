"""One-hot encoding of experimental context columns for LNPDB data.

Context columns capture the experimental setup (cell type, target organ,
route of administration, cargo, measurement method, batching design).
These are critical covariates: the same formulation can have very different
Experiment_value depending on the assay system.
"""


import pandas as pd

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
