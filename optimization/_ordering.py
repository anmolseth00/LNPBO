"""Column-ordering utilities for aligning dataframes with optimizer state."""


def order_df_columns(config, pbounds):
    """Return dataframe column names in the order the optimizer expects.

    The optimizer sorts parameters alphabetically by name.  This function
    walks that sorted order and expands each parameter into its constituent
    dataframe columns, producing a flat list suitable for indexing a
    ``pd.DataFrame``.

    Args:
        config: Space configuration dict with a ``"parameters"`` list,
            where each entry has ``"name"`` and ``"columns"`` keys.
        pbounds: Dict whose keys are parameter names (used only for
            iteration order via ``sorted()``).

    Returns:
        List of column name strings in optimizer-compatible order.
    """
    res = []
    for parameter in sorted(pbounds.keys()):
        for p in config["parameters"]:
            if p["name"] == parameter:
                res.extend(p["columns"])
    return res
