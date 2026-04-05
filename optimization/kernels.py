"""Custom GP kernels for molecular fingerprint and compositional spaces.

Provides:
    TanimotoKernel             — for molecular fingerprints (binary/count)
    AitchisonKernel            — for simplex-constrained molar ratios
    CompositionalProductKernel — product kernel with domain-specific sub-kernels
"""

import torch
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel


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
        """Compute the Tanimoto kernel matrix.

        k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

        For diagonal entries, returns ones (self-similarity = 1). The
        denominator is clamped to avoid division by zero for zero vectors.

        Args:
            x1: Tensor of shape (..., N, D).
            x2: Tensor of shape (..., M, D).
            diag: If True, return only the diagonal (N,) instead of
                the full (N, M) kernel matrix.
            last_dim_is_batch: GPyTorch batch mode flag (unused).

        Returns:
            Kernel matrix of shape (..., N, M) or diagonal (..., N).
        """
        if diag:
            return torch.ones(
                *x1.shape[:-2],
                x1.shape[-2],
                dtype=x1.dtype,
                device=x1.device,
            )

        dot_prod = torch.matmul(x1, x2.transpose(-1, -2))
        x1_sq = (x1**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2**2).sum(dim=-1, keepdim=True)

        denom = x1_sq + x2_sq.transpose(-1, -2) - dot_prod
        return (dot_prod / denom.clamp_min(1e-6)).clamp_min(0)


class AitchisonKernel(Kernel):
    """Aitchison-geometry kernel for compositional (simplex) data.

    Computes a Gaussian kernel using Aitchison distance, which is the
    natural metric on the simplex. Inputs should be positive compositions
    (e.g., molar ratios) that will be internally log-transformed.

    k(x, y) = exp(-d_A(x, y)^2 / (2 * lengthscale^2))

    where d_A(x, y) = ||clr(x) - clr(y)|| and clr is the centered
    log-ratio transform: clr(x)_i = log(x_i / geometric_mean(x)).

    This kernel respects the geometry of the simplex: compositions that
    are proportionally similar (e.g., 50:30:20 and 25:15:10) have
    distance zero, which Euclidean kernels on raw ratios do not capture.

    Parameters
    ----------
    eps : float
        Small constant added to avoid log(0) for zero compositions.

    References
    ----------
    Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
        Chapman & Hall.
    Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
        Compositional Data Analysis." Mathematical Geology, 35(3), 279-300.
    Pawlowsky-Glahn, V. & Buccianti, A. (2011). "Compositional Data
        Analysis: Theory and Applications." Wiley.
    """

    has_lengthscale = True

    def __init__(self, eps=1e-6, **kwargs):
        """Initialize the Aitchison kernel.

        Args:
            eps: Small constant added before log to avoid log(0).
            **kwargs: Passed to gpytorch.kernels.Kernel.
        """
        super().__init__(**kwargs)
        self.eps = eps

    def _clr(self, x):
        """Apply the centered log-ratio (CLR) transform.

        clr(x)_i = log(x_i) - mean(log(x))

        Maps compositional data from the simplex to unconstrained
        Euclidean space while preserving compositional relationships.

        Args:
            x: Tensor of positive compositions, shape (..., D).

        Returns:
            CLR-transformed tensor of the same shape.

        Reference:
            Aitchison, J. (1986). "The Statistical Analysis of
            Compositional Data." Chapman & Hall.
        """
        x = x.clamp(min=self.eps)
        log_x = torch.log(x)
        geo_mean = log_x.mean(dim=-1, keepdim=True)
        return log_x - geo_mean

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Compute the Aitchison-geometry Gaussian kernel matrix.

        k(x, y) = exp(-d_A(x, y)^2 / (2 * lengthscale^2))

        where d_A is the Aitchison distance computed via CLR transform.
        Supports both isotropic and ARD lengthscales.

        Args:
            x1: Tensor of positive compositions, shape (..., N, D).
            x2: Tensor of positive compositions, shape (..., M, D).
            diag: If True, return diagonal (..., N) only.
            last_dim_is_batch: GPyTorch batch mode flag (unused).

        Returns:
            Kernel matrix of shape (..., N, M) or diagonal (..., N).

        Reference:
            Aitchison, J. (1986). "The Statistical Analysis of
            Compositional Data." Chapman & Hall, Ch. 4.
            Egozcue, J.J. et al. (2003). "Isometric Logratio
            Transformations for Compositional Data Analysis."
            Mathematical Geology, 35(3), 279-300.
        """
        clr1 = self._clr(x1)
        clr2 = self._clr(x2)

        # Scale by lengthscale (ARD or isotropic)
        clr1 = clr1 / self.lengthscale
        clr2 = clr2 / self.lengthscale

        if diag:
            diff = clr1 - clr2
            return torch.exp(-0.5 * (diff**2).sum(dim=-1))

        # Squared Aitchison distance via expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        sq1 = (clr1**2).sum(dim=-1, keepdim=True)
        sq2 = (clr2**2).sum(dim=-1, keepdim=True)
        cross = torch.matmul(clr1, clr2.transpose(-1, -2))
        sq_dist = sq1 + sq2.transpose(-1, -2) - 2 * cross
        return torch.exp(-0.5 * sq_dist.clamp_min(0))


class CompositionalProductKernel(Kernel):
    """Product kernel with domain-specific sub-kernels for LNP formulations.

    Decomposes the input feature vector into three groups by column index:
    1. Molecular structure features -> Matern-5/2 kernel with ARD
    2. Compositional ratio features -> Aitchison kernel (CLR transform)
    3. Remaining continuous features -> Matern-5/2 kernel with ARD

    The product structure assumes conditional independence between
    molecular identity, composition, and process parameters:

        k(x, x') = k_struct(x_fp, x'_fp) * k_aitchison(x_ratio, x'_ratio) * k_synth(x_synth, x'_synth)

    Any subset of feature groups may be empty; the corresponding sub-kernel
    is simply omitted (the product of the remaining sub-kernels is used).

    Designed for PCA-reduced molecular descriptors (e.g., LANTERN encoding).
    Uses Matern(ARD) for structure features since PCA projections live in
    Euclidean space, not the non-negative count-vector space required by
    Tanimoto. The Aitchison sub-kernel handles compositional ratios on the
    simplex via centered log-ratio (CLR) transform.

    Parameters
    ----------
    fp_indices : list[int]
        Column indices for molecular structure features (PCA-reduced).
    ratio_indices : list[int]
        Column indices for compositional ratio features (simplex data).
    synth_indices : list[int]
        Column indices for synthesis/process parameters (Euclidean).

    References
    ----------
    Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
        Chapman & Hall.
    Duvenaud, D. et al. (2011). "Additive Gaussian Processes." NeurIPS.
        (discusses product/additive kernel decompositions)
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, fp_indices, ratio_indices, synth_indices, **kwargs):
        """Initialize the compositional product kernel with index groups.

        Creates sub-kernels for each non-empty feature group:
        - Structure (fp_indices): ScaleKernel(Matern-5/2 with ARD)
        - Composition (ratio_indices): ScaleKernel(AitchisonKernel)
        - Synthesis (synth_indices): ScaleKernel(Matern-5/2 with ARD)

        Args:
            fp_indices: Column indices for molecular structure features.
            ratio_indices: Column indices for compositional ratios.
            synth_indices: Column indices for synthesis/process parameters.
            **kwargs: Passed to gpytorch.kernels.Kernel.
        """
        super().__init__(**kwargs)
        self.register_buffer("fp_idx", torch.tensor(fp_indices, dtype=torch.long))
        self.register_buffer("ratio_idx", torch.tensor(ratio_indices, dtype=torch.long))
        self.register_buffer("synth_idx", torch.tensor(synth_indices, dtype=torch.long))

        self._sub_kernel_dict = torch.nn.ModuleDict()
        if len(fp_indices) > 0:
            self._sub_kernel_dict["structure"] = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=len(fp_indices)))
        if len(ratio_indices) > 0:
            self._sub_kernel_dict["aitchison"] = ScaleKernel(AitchisonKernel())
        if len(synth_indices) > 0:
            self._sub_kernel_dict["matern"] = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=len(synth_indices),
                )
            )

    def _slice(self, x, idx):
        """Extract columns from input tensor by index buffer.

        Args:
            x: Input tensor of shape (..., D).
            idx: Long tensor of column indices to select.

        Returns:
            Sliced tensor of shape (..., len(idx)).
        """
        return x.index_select(-1, idx)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Compute the product of domain-specific sub-kernels.

        Slices input tensors by registered index buffers, evaluates each
        active sub-kernel on its feature group, and returns their product.
        Sub-kernels with empty index lists are skipped. If no sub-kernels
        are active, returns a matrix of ones.

        Args:
            x1: Tensor of shape (..., N, D).
            x2: Tensor of shape (..., M, D).
            diag: If True, return diagonal (..., N) only.
            last_dim_is_batch: GPyTorch batch mode flag (unused).

        Returns:
            Product kernel matrix of shape (..., N, M) or diagonal (..., N).
        """
        result = None
        for name, idx_buf in [
            ("structure", self.fp_idx),
            ("aitchison", self.ratio_idx),
            ("matern", self.synth_idx),
        ]:
            if name not in self._sub_kernel_dict or len(idx_buf) == 0:
                continue
            sub = self._sub_kernel_dict[name]
            s1 = self._slice(x1, idx_buf)
            s2 = self._slice(x2, idx_buf)
            k = sub.forward(s1, s2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
            result = k if result is None else result * k

        if result is None:
            # No sub-kernels active: return ones (degenerate case)
            if diag:
                return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            return torch.ones(*x1.shape[:-2], x1.shape[-2], x2.shape[-2], dtype=x1.dtype, device=x1.device)
        return result
