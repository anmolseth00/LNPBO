"""Kernel and GP definitions for exact FSBO."""

from __future__ import annotations

import gpytorch
import torch


def make_fsbo_kernel(
    *,
    base_kernel: str,
    feature_dim: int,
    num_mixtures: int,
) -> gpytorch.kernels.Kernel:
    if base_kernel == "rbf":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )
    if base_kernel == "spectral":
        return gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=feature_dim,
        )
    raise ValueError(f"Unsupported base kernel: {base_kernel}")


class DeepKernelExactGP(gpytorch.models.ExactGP):
    """Exact GP with a learned feature extractor."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        feature_extractor: torch.nn.Module,
        base_kernel: str = "rbf",
        num_mixtures: int = 4,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        if hasattr(feature_extractor, "output_dim"):
            feature_dim = int(feature_extractor.output_dim)
        else:
            feature_dim = int(train_x.shape[-1])
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = make_fsbo_kernel(
            base_kernel=base_kernel,
            feature_dim=feature_dim,
            num_mixtures=num_mixtures,
        )
        self.base_kernel = base_kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        z = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z),
            self.covar_module(z),
        )

    def initialize_kernel_from_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        if self.base_kernel != "spectral":
            return
        kernel = self.covar_module
        if isinstance(kernel, gpytorch.kernels.ScaleKernel):
            kernel = kernel.base_kernel
        if hasattr(kernel, "initialize_from_data"):
            with torch.no_grad():
                kernel.initialize_from_data(self.feature_extractor(x_batch), y_batch)
