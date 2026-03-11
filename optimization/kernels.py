"""Custom GP kernels for molecular fingerprint spaces."""

import torch
from gpytorch.kernels import Kernel


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) kernel for molecular fingerprints.

    k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    Valid positive definite kernel on all of R^d (Tripp et al., NeurIPS 2023).
    Appropriate for binary and count Morgan fingerprints where Euclidean
    distance is not meaningful.

    References
    ----------
    Ralaivola, L., Swamidass, S. J., Saigo, H., & Baldi, P.
        "Graph kernels for chemical informatics." Neural Networks 18(8), 2005.
    Tripp, A., Bacallado, S., Singh, S., & Hernandez-Lobato, J. M.
        "Tanimoto Random Features for Scalable Molecular Machine Learning."
        NeurIPS 2023. arXiv:2306.14809.
    """

    is_stationary = False
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag:
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device,
            )

        dot_prod = torch.matmul(x1, x2.transpose(-1, -2))
        x1_sq = (x1**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2**2).sum(dim=-1, keepdim=True)

        denom = x1_sq + x2_sq.transpose(-1, -2) - dot_prod
        return (dot_prod / denom.clamp_min(1e-6)).clamp_min(0)
