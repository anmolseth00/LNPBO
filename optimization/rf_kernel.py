"""Random Forest proximity kernel for GPyTorch.

Defines a GP kernel where K(x_i, x_j) equals the fraction of trees in a
fitted Random Forest for which x_i and x_j fall into the same leaf node.
This is a valid positive semi-definite kernel that inherits the RF's
data-adaptive feature weighting, automatic interaction detection, and
native handling of mixed categorical/continuous inputs.

The kernel matrix is computed efficiently by chunking the tree ensemble
into groups and vectorizing the leaf-agreement comparison via NumPy
broadcasting, then converting the result to a torch Tensor.

Reference
---------
Scornet, E. (2016). "Random Forests and Kernel Methods."
    IEEE Transactions on Information Theory, 62(3), 1485-1500.
    DOI: 10.1109/TIT.2015.2507602

Breiman, L. (2001). "Random Forests."
    Machine Learning, 45(1), 5-32.
"""

from __future__ import annotations

import numpy as np
import torch
from gpytorch.kernels import Kernel


class RandomForestKernel(Kernel):
    """GP kernel derived from Random Forest leaf co-occurrence proximity.

    For a fitted RF with T trees, the kernel is:

        K(x_i, x_j) = (1/T) * sum_{t=1}^{T} I[leaf_t(x_i) == leaf_t(x_j)]

    This kernel is:
    - Positive semi-definite (it is an inner product in the binary
      leaf-indicator feature space; see Scornet 2016, Theorem 1).
    - Data-adaptive: splits are learned from (X, Y), so the kernel
      automatically captures relevant interactions and feature scales.
    - Suitable for mixed feature spaces: RF handles continuous and
      categorical splits natively through its greedy partitioning.

    The kernel does NOT have a learnable lengthscale — the RF's
    splitting decisions serve an analogous role.

    Parameters
    ----------
    train_X : torch.Tensor
        (N, D) training features used to fit the Random Forest.
    train_Y : torch.Tensor
        (N, 1) or (N,) training targets.
    n_estimators : int
        Number of trees in the forest.
    max_features : float or str
        Fraction of features considered per split. Passed to
        ``RandomForestRegressor(max_features=...)``.
    min_samples_leaf : int
        Minimum samples per leaf. Controls kernel granularity:
        smaller values give a more expressive (but noisier) kernel.
    random_state : int
        RNG seed for the RF.

    Reference
    ---------
    Scornet, E. (2016). "Random Forests and Kernel Methods."
        IEEE Transactions on Information Theory, 62(3), 1485-1500.
    """

    is_stationary = False
    has_lengthscale = False
    _CHUNK = 50

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        n_estimators: int = 500,
        max_features: float | str = 1.0,
        min_samples_leaf: int = 3,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from sklearn.ensemble import RandomForestRegressor

        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        X_np = train_X.detach().cpu().numpy()
        Y_np = train_Y.detach().cpu().numpy().ravel()
        self.rf.fit(X_np, Y_np)

        # Cache training leaves to avoid recomputing on every forward() call
        self._train_leaves = self.rf.apply(X_np)

    def _get_leaves(self, X: torch.Tensor) -> np.ndarray:
        """Return leaf indices (n, n_trees) for input points."""
        X_np = X.detach().cpu().numpy()
        return self.rf.apply(X_np)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        """Compute the RF proximity kernel matrix.

        Uses chunked NumPy broadcasting to avoid both the O(T) Python
        loop and the O(n1 * n2 * T) memory spike from full expansion.
        Trees are processed in chunks of ``_CHUNK`` (default 50), each
        chunk requiring O(n1 * n2 * chunk) temporary memory.
        """
        if diag:
            return torch.ones(x1.shape[0], dtype=x1.dtype, device=x1.device)

        leaves1 = self._get_leaves(x1)  # (n1, T)
        leaves2 = self._get_leaves(x2)  # (n2, T)
        n_trees = leaves1.shape[1]

        K_np = np.zeros((leaves1.shape[0], leaves2.shape[0]), dtype=np.float64)
        for start in range(0, n_trees, self._CHUNK):
            end = min(start + self._CHUNK, n_trees)
            # (n1, chunk, 1) == (1, chunk, n2) -> (n1, chunk, n2)
            eq = leaves1[:, start:end, np.newaxis] == leaves2[:, start:end].T[np.newaxis, :, :]
            K_np += eq.sum(axis=1)
        K_np /= n_trees

        return torch.tensor(K_np, dtype=x1.dtype, device=x1.device)
