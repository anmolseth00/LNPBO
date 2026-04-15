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
        """Return leaf indices for input points.

        Accepts either 2-D tensors of shape ``(n, d)`` or tensors with
        leading batch dimensions of shape ``(..., n, d)`` and preserves
        those leading dimensions in the returned leaf array.
        """
        X_np = X.detach().cpu().numpy()
        if X_np.ndim < 2:
            raise ValueError("RandomForestKernel expects inputs with at least 2 dimensions")
        flat = X_np.reshape(-1, X_np.shape[-1])
        leaves = self.rf.apply(flat)
        return leaves.reshape(*X_np.shape[:-1], -1)

    def _kernel_from_leaves(self, leaves1: np.ndarray, leaves2: np.ndarray, dtype, device) -> torch.Tensor:
        """Compute proximities from precomputed leaf indices."""
        squeeze_batch = False
        if leaves1.ndim == 2:
            leaves1 = leaves1[None, ...]
            leaves2 = leaves2[None, ...]
            squeeze_batch = True

        n_trees = leaves1.shape[-1]
        K_np = np.zeros((leaves1.shape[0], leaves1.shape[1], leaves2.shape[1]), dtype=np.float64)
        for start in range(0, n_trees, self._CHUNK):
            end = min(start + self._CHUNK, n_trees)
            # (B, N, C, 1) == (B, 1, C, M) -> (B, N, C, M)
            lhs = leaves1[:, :, start:end][:, :, :, None]
            rhs = np.transpose(leaves2[:, :, start:end], (0, 2, 1))[:, None, :, :]
            K_np += (lhs == rhs).sum(axis=2)
        K_np /= n_trees
        if squeeze_batch:
            K_np = K_np[0]
        return torch.tensor(K_np, dtype=dtype, device=device)

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
        batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
        if diag:
            return torch.ones(*batch_shape, x1.shape[-2], dtype=x1.dtype, device=x1.device)

        if x1.dim() == 2 and x2.dim() == 2:
            return self._kernel_from_leaves(self._get_leaves(x1), self._get_leaves(x2), x1.dtype, x1.device)

        x1_exp = x1.expand(*batch_shape, *x1.shape[-2:])
        x2_exp = x2.expand(*batch_shape, *x2.shape[-2:])

        leaves1 = self._get_leaves(x1_exp)
        leaves2 = self._get_leaves(x2_exp)
        return self._kernel_from_leaves(leaves1, leaves2, x1.dtype, x1.device)
