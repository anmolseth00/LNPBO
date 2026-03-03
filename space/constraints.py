from __future__ import annotations
import numpy as np

def validate_mixture(
    values,
    bounds,
    target_sum,
    atol=1e-6,
):
    """
    Validate mixture feasibility.
    """
    values = np.array(values)

    if np.any(values < bounds[:, 0]):
        return False

    if np.any(values > bounds[:, 1]):
        return False

    if not np.isclose(values.sum(), target_sum, atol=atol):
        return False

    return True
