"""Spectral-Normalized Neural GP (SNGP) surrogate.

SNGP modifies a standard MLP with:
  (a) spectral normalization on hidden layers to enforce distance-awareness
  (b) a Random Fourier Feature GP output layer (Rahimi & Recht 2007)

Distance-awareness ensures that inputs far from training data receive high
predictive uncertainty, which is critical for BO exploration.

References:
    Liu, J.Z. et al. (2023). "A Simple Approach to Improve Single-Model Deep
        Uncertainty via Distance-Awareness." JMLR 24(42), 1-63.
    Rahimi, A. & Recht, B. (2007). "Random Features for Large-Scale Kernel
        Machines." NeurIPS 2007.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for RBF kernel approximation.

    Reference: Rahimi & Recht (2007), NeurIPS 2007.
    """

    def __init__(self, in_dim, n_features=1024, lengthscale=1.0):
        super().__init__()
        self.register_buffer("W", torch.randn(in_dim, n_features) / lengthscale)
        self.register_buffer("b", torch.rand(n_features) * 2 * math.pi)
        self.n_features = n_features

    def forward(self, x):
        z = x @ self.W + self.b
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)


class SNGP(nn.Module):
    """Spectral-Normalized Neural GP.

    Architecture:
      1. Spectral-normalized hidden layers (Lipschitz constraint).
      2. Random Fourier Feature layer approximates an RBF kernel.
      3. Laplace approximation on the output layer gives posterior variance.

    Reference: Liu et al. (2023), JMLR 24(42).
    """

    def __init__(self, input_dim, hidden_dims=(256, 128), n_random_features=1024,
                 lengthscale=1.0, ridge_penalty=1.0):
        super().__init__()
        self.n_random_features = n_random_features
        self.ridge_penalty = ridge_penalty

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(spectral_norm(nn.Linear(prev_dim, h_dim)))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.rff = RandomFourierFeatures(hidden_dims[-1], n_random_features, lengthscale)
        self.output_layer = nn.Linear(n_random_features, 1, bias=True)

        self.register_buffer(
            "precision", torch.eye(n_random_features) * ridge_penalty
        )
        self.register_buffer("n_train", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        h = self.backbone(x)
        phi = self.rff(h)
        return self.output_layer(phi).squeeze(-1)

    def reset_precision(self):
        D = self.n_random_features
        self.precision.copy_(torch.eye(D, device=self.precision.device) * self.ridge_penalty)
        self.n_train.zero_()

    def update_precision(self, x):
        """Update precision matrix with a batch of training features."""
        with torch.no_grad():
            h = self.backbone(x)
            phi = self.rff(h)
            self.precision.add_(phi.T @ phi)
            self.n_train.add_(phi.size(0))

    def predict_with_uncertainty(self, x):
        """Predict mean and std using Laplace approximation on the GP layer."""
        self.eval()
        with torch.no_grad():
            h = self.backbone(x)
            phi = self.rff(h)
            mu = self.output_layer(phi).squeeze(-1)
            try:
                L = torch.linalg.cholesky(self.precision)
                phi_solve = torch.linalg.solve_triangular(L, phi.T, upper=False)
                var = (phi_solve ** 2).sum(dim=0)
            except torch.linalg.LinAlgError:
                cov = torch.linalg.solve(self.precision, phi.T)
                var = (phi * cov.T).sum(dim=-1)
        return mu.cpu().numpy(), var.sqrt().cpu().numpy()


def train_sngp(X_train, y_train, input_dim, hidden_dims=(256, 128),
               n_random_features=1024, epochs=100, lr=1e-3, batch_size=256,
               ridge_penalty=1.0, lengthscale=1.0):
    """Fit an SNGP model and build the precision matrix for posterior inference.

    Parameters
    ----------
    X_train : Tensor of shape (N, D).
    y_train : Tensor of shape (N,).
    input_dim : Number of input features.

    Returns
    -------
    Fitted SNGP model with precision matrix ready for predict_with_uncertainty().
    """
    model = SNGP(input_dim, hidden_dims, n_random_features, lengthscale, ridge_penalty)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

    model.reset_precision()
    for (xb,) in DataLoader(TensorDataset(X_train), batch_size=1024, shuffle=False):
        model.update_precision(xb)

    return model
