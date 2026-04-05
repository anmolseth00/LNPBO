"""Post-hoc Laplace approximation on MLP surrogate for uncertainty quantification.

Computes a Laplace approximation to the posterior over weights of a trained
MLP, yielding a Gaussian predictive distribution. Supports last-layer and
all-weights variants.

If laplace-torch is available, uses it directly. Otherwise, provides a manual
last-layer Laplace implementation using Kronecker-factored (KFAC) or diagonal
Hessian approximation.

References:
    Daxberger, E. et al. (2021). "Laplace Redux -- Effortless Bayesian Deep
        Learning." NeurIPS 2021. arXiv:2106.14806.
    MacKay, D.J.C. (1992). "A Practical Bayesian Framework for Backpropagation
        Networks." Neural Computation 4(3).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .surrogate_mlp import SurrogateMLP


def _check_laplace_available():
    try:
        import laplace  # noqa: F401
        return True
    except ImportError:
        return False


LAPLACE_AVAILABLE = _check_laplace_available()


def train_mlp(model, X_train, y_train, epochs=100, lr=1e-3, batch_size=256):
    """Train an MLP on (X, y) with MSE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


class ManualLastLayerLaplace:
    """Manual last-layer Laplace approximation for SurrogateMLP.

    Computes the posterior N(w_MAP, (H + alpha*I)^{-1}) over last-layer
    weights, where H is the Hessian of the loss w.r.t. last-layer parameters.

    Reference: MacKay (1992), Daxberger et al. (2021).
    """

    def __init__(self, model, hessian_structure="kron"):
        self.model = model
        self.hessian_structure = hessian_structure
        self.prior_precision = 1.0
        self.sigma_obs = 1.0
        self._fitted = False

    def _extract_features(self, X):
        with torch.no_grad():
            h = torch.relu(self.model.fc1(X))
            h = torch.relu(self.model.fc2(h))
        return h

    def fit(self, train_loader):
        """Compute the Hessian of the last layer using training data."""
        all_features, all_targets = [], []
        for xb, yb in train_loader:
            all_features.append(self._extract_features(xb))
            all_targets.append(yb)

        Phi = torch.cat(all_features, dim=0)
        y = torch.cat(all_targets, dim=0)
        n = Phi.size(0)

        with torch.no_grad():
            all_x = torch.cat([xb for xb, _ in train_loader], dim=0)
            pred = self.model(all_x)
        residuals = y - pred[:n]
        self.sigma_obs = float(residuals.var().sqrt().clamp(min=1e-6))

        if self.hessian_structure == "diag":
            diag = (Phi ** 2).sum(dim=0) / (self.sigma_obs ** 2)
            bias_diag = torch.tensor([n / (self.sigma_obs ** 2)])
            self._diag_H = torch.cat([diag, bias_diag])
        elif self.hessian_structure == "kron":
            Phi_aug = torch.cat([Phi, torch.ones(n, 1)], dim=1)
            self._H = (Phi_aug.T @ Phi_aug) / (self.sigma_obs ** 2)
        self._Phi_train = Phi
        self._fitted = True

    def optimize_prior_precision(self, method="marginal_likelihood"):
        """Optimize prior precision via type-II maximum likelihood (grid search)."""
        candidates = np.logspace(-3, 3, 20)
        best_prec, best_score = 1.0, -np.inf
        d_plus_1 = self._Phi_train.size(1) + 1

        for prec in candidates:
            if self.hessian_structure == "diag":
                posterior_diag = self._diag_H + prec
                log_det = posterior_diag.log().sum()
                score = 0.5 * d_plus_1 * np.log(prec) - 0.5 * float(log_det)
            elif self.hessian_structure == "kron":
                M = self._H + prec * torch.eye(d_plus_1)
                try:
                    L = torch.linalg.cholesky(M)
                    log_det = 2.0 * L.diagonal().log().sum()
                except torch.linalg.LinAlgError:
                    continue
                score = 0.5 * d_plus_1 * np.log(prec) - 0.5 * float(log_det)
            if score > best_score:
                best_score = score
                best_prec = prec
        self.prior_precision = float(best_prec)

    def predict(self, X):
        """Predict (mu, var) for new inputs."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")
        with torch.no_grad():
            mu = self.model(X)
            Phi = self._extract_features(X)

        n_test = Phi.size(0)
        d = Phi.size(1)
        Phi_aug = torch.cat([Phi, torch.ones(n_test, 1)], dim=1)

        if self.hessian_structure == "diag":
            posterior_diag = self._diag_H + self.prior_precision
            var_epistemic = (Phi_aug ** 2 / posterior_diag.unsqueeze(0)).sum(dim=1)
        elif self.hessian_structure == "kron":
            M = self._H + self.prior_precision * torch.eye(d + 1)
            try:
                L = torch.linalg.cholesky(M)
                phi_solve = torch.linalg.solve_triangular(L, Phi_aug.T, upper=False)
                var_epistemic = (phi_solve ** 2).sum(dim=0)
            except torch.linalg.LinAlgError:
                cov = torch.linalg.solve(M, Phi_aug.T)
                var_epistemic = (Phi_aug * cov.T).sum(dim=1)

        var = self.sigma_obs ** 2 + var_epistemic
        return mu.cpu().numpy(), var.cpu().numpy()


class LaplaceTorchWrapper:
    """Wrapper for the laplace-torch library (Daxberger et al. 2021)."""

    def __init__(self, model, subset_of_weights="last_layer", hessian_structure="kron"):
        from laplace import Laplace
        self.la = Laplace(
            model, "regression",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
        )

    def fit(self, train_loader):
        self.la.fit(train_loader)

    def optimize_prior_precision(self, method="marginal_likelihood"):
        self.la.optimize_prior_precision(method=method)

    def predict(self, X):
        f_mu, f_var = self.la(X)
        return f_mu.squeeze(-1).cpu().numpy(), f_var.squeeze(-1).cpu().numpy()


def build_laplace(model, subset_of_weights="last_layer", hessian_structure="kron"):
    """Build a Laplace approximation wrapper, preferring laplace-torch if available."""
    if LAPLACE_AVAILABLE and subset_of_weights == "last_layer":
        try:
            return LaplaceTorchWrapper(model, subset_of_weights, hessian_structure)
        except (RuntimeError,):
            pass
    if subset_of_weights == "last_layer":
        return ManualLastLayerLaplace(model, hessian_structure)
    if LAPLACE_AVAILABLE:
        return LaplaceTorchWrapper(model, subset_of_weights, hessian_structure)
    raise ValueError(
        f"subset_of_weights='{subset_of_weights}' with all weights requires "
        "laplace-torch. Install with: pip install laplace-torch"
    )
