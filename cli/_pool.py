"""Shared candidate pool construction logic for CLI and pipeline."""

import pandas as pd

from ..data.dataset import Dataset


def build_candidate_pool(
    encoded,
    surrogate_type,
    pool_csv=None,
    feature_type=None,
    reduction=None,
):
    """Build a candidate pool DataFrame for pool-based surrogates.

    Args:
        encoded: Encoded Dataset object (training data).
        surrogate_type: Surrogate type string. Returns None for "gp_sklearn"
            (continuous optimization doesn't need a discrete pool).
        pool_csv: Path to external pool CSV file, or None to use training data.
        feature_type: Feature type string (for encoding external pool).
        reduction: Reduction method (for encoding external pool).

    Returns:
        DataFrame of encoded candidate pool, or None for sklearn GP.
    """
    if surrogate_type == "gp_sklearn":
        return None

    if pool_csv:
        pool_df = pd.read_csv(pool_csv)
        pool_dataset = Dataset(pool_df, source="lnpdb", name="candidate_pool")
        pool_encoded = pool_dataset.encode_dataset(
            feature_type=feature_type,
            reduction=reduction,
            fitted_transformers_in=encoded.fitted_transformers,
        )
        return pool_encoded.df

    return encoded.df
