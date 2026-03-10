"""Target normalization utilities shared between Optimizer and benchmark runner."""

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


def copula_transform(values, x_new=None):
    """Rank-based copula transform to standard normal.

    When ``x_new`` is provided, the transform is fitted on ``values``
    and applied to ``x_new`` (each new value is ranked relative to the
    reference distribution).  This avoids scale mismatch when the model
    was trained on the copula of ``values`` alone.

    Reference: quantile normalization / probability integral transform.
    """
    if x_new is None:
        n = len(values)
        ranks = pd.Series(values).rank(method="average")
        u = (ranks - 0.5) / n
        return _norm.ppf(u)

    ref = np.sort(values)
    n = len(ref)
    ranks = np.searchsorted(ref, x_new, side="right").astype(float)
    u = np.clip((ranks + 0.5) / (n + 1), 1e-10, 1 - 1e-10)
    return _norm.ppf(u)


def normalize_values(y, method):
    """Normalize a numpy array of target values.

    Returns a new array (does not modify in place).
    """
    if method == "none":
        return y
    if method == "copula":
        return copula_transform(y)
    if method == "zscore":
        mu, sigma = y.mean(), y.std()
        if sigma > 0:
            return (y - mu) / sigma
    return y


def normalize_targets(df, method):
    """Apply target normalization to a dataframe in-place."""
    if method == "none":
        return
    col = "Experiment_value"
    df[col] = normalize_values(df[col].values, method)
