"""Custom GP kernels for molecular fingerprint and compositional spaces.

Provides:
    TanimotoKernel             - for molecular fingerprints (binary/count)
    AitchisonKernel            - for simplex-constrained molar ratios
    CompositionalProductKernel - product kernel with domain-specific sub-kernels
"""

import torch
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) kernel for molecular fingerprints.

    k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    This is the continuous (dot-product) Tanimoto kernel, which Tripp et al.
    (NeurIPS 2023, Thm. 4.1) prove is positive definite on all of R^d -- not
    only on non-negative inputs. On binary/count Morgan fingerprints it reduces
    to the standard Jaccard/MinMax similarity (Ralaivola et al. 2005). The
    forward pass clamps the denominator and any (legitimately negative)
    off-diagonal entries for numerical safety.

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
        """Tanimoto kernel matrix; denominator clamped to avoid div-by-zero."""
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
        """eps is added before log to avoid log(0)."""
        super().__init__(**kwargs)
        self.eps = eps

    def _clr(self, x):
        """Centered log-ratio transform: clr(x)_i = log(x_i) - mean(log(x))."""
        x = x.clamp(min=self.eps)
        log_x = torch.log(x)
        log_geo_mean = log_x.mean(dim=-1, keepdim=True)
        return log_x - log_geo_mean

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Aitchison-geometry Gaussian kernel (CLR distance), ARD or isotropic."""
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
    Rasmussen, C.E. & Williams, C.K.I. (2006). "Gaussian Processes for
        Machine Learning," Sec. 4.2.4 (products of kernels over disjoint
        input subspaces).
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, fp_indices, ratio_indices, synth_indices, **kwargs):
        """Build a ScaleKernel sub-kernel per non-empty feature group."""
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
        return x.index_select(-1, idx)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Product of active sub-kernels over their feature groups (ones if none)."""
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
