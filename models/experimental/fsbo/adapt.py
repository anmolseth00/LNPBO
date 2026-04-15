"""Target-task adaptation and predictive utilities for exact FSBO."""

from __future__ import annotations

import gpytorch
import numpy as np
import torch
from scipy.stats import norm

from .encoder import FSBOFeatureExtractor, IdentityFeatures, to_tensor
from .kernel import DeepKernelExactGP
from .meta_train import FSBOMetaState


def build_meta_initialized_model(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    meta_state: FSBOMetaState,
) -> tuple[DeepKernelExactGP, gpytorch.likelihoods.GaussianLikelihood]:
    train_x = to_tensor(X_obs)
    train_y = torch.tensor(np.asarray(y_obs, dtype=np.float64), dtype=torch.float64)
    feature_extractor = FSBOFeatureExtractor(train_x.shape[-1], meta_state.hidden_dims).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = DeepKernelExactGP(
        train_x,
        train_y,
        likelihood,
        feature_extractor=feature_extractor,
        base_kernel=meta_state.base_kernel,
        num_mixtures=meta_state.num_mixtures,
    ).double()
    model.feature_extractor.load_state_dict(meta_state.feature_extractor_state)
    model.mean_module.load_state_dict(meta_state.mean_state)
    model.covar_module.load_state_dict(meta_state.covar_state)
    likelihood.load_state_dict(meta_state.likelihood_state)
    model.eval()
    likelihood.eval()
    return model, likelihood


def build_cold_gp_model(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
) -> tuple[DeepKernelExactGP, gpytorch.likelihoods.GaussianLikelihood]:
    train_x = to_tensor(X_obs)
    train_y = torch.tensor(np.asarray(y_obs, dtype=np.float64), dtype=torch.float64)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = DeepKernelExactGP(
        train_x,
        train_y,
        likelihood,
        feature_extractor=IdentityFeatures(),
        base_kernel="rbf",
    ).double()
    model.eval()
    likelihood.eval()
    return model, likelihood


def optimize_gp_model(
    model: DeepKernelExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    n_steps: int,
    lr: float,
) -> list[float]:
    train_x = to_tensor(X_obs)
    train_y = torch.tensor(np.asarray(y_obs, dtype=np.float64), dtype=torch.float64)
    model.set_train_data(train_x, train_y, strict=False)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses: list[float] = []
    for _ in range(max(n_steps, 0)):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    likelihood.eval()
    return losses


def predict_mean_std(
    model: DeepKernelExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    x_t = to_tensor(X)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(x_t))
        mean = posterior.mean.detach().cpu().numpy().astype(np.float64)
        std = posterior.stddev.detach().cpu().numpy().astype(np.float64)
    return mean, np.maximum(std, 1e-12)


def expected_improvement(mean: np.ndarray, std: np.ndarray, best_f: float) -> np.ndarray:
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
    improvement = np.asarray(mean, dtype=np.float64) - float(best_f)
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)
