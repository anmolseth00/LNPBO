"""Deep Ensemble surrogate for uncertainty quantification.

Trains M independent neural networks with different random initializations
and optional bootstrap sampling. Uncertainty is estimated from ensemble
disagreement (standard deviation of member predictions).

Reference:
    Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017).
    "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
    NeurIPS 2017. arXiv:1612.01474.
"""

import numpy as np
import torch
import torch.nn.functional as F


from .surrogate_mlp import SurrogateMLP


class DeepEnsemble:
    """Deep ensemble of independently trained MLPs.

    Reference:
        Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017).
        "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
        NeurIPS 2017. arXiv:1612.01474.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_models : int
        Number of ensemble members (M).
    epochs : int
        Training epochs per member.
    lr : float
        Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        input_dim: int,
        n_models: int = 5,
        hidden_dims: tuple[int, ...] = (256, 128),
        epochs: int = 100,
        lr: float = 1e-3,
    ):
        self.input_dim = input_dim
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.models: list[SurrogateMLP] = []

    def fit(self, X: np.ndarray, y: np.ndarray, bootstrap: bool = True, seed: int = 42):
        """Train each ensemble member independently.

        Each member gets a different random initialization (via distinct seed).
        If bootstrap=True, each member also trains on a different bootstrap
        sample of the data.
        """
        self.models = []
        rng = np.random.RandomState(seed)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for _m in range(self.n_models):
            member_seed = rng.randint(0, 2**31)
            torch.manual_seed(member_seed)
            model = SurrogateMLP(self.input_dim)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)

            if bootstrap:
                boot_rng = np.random.RandomState(member_seed)
                idx = boot_rng.choice(len(X), size=len(X), replace=True)
                X_b = X_t[idx]
                y_b = y_t[idx]
            else:
                X_b = X_t
                y_b = y_t

            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = model(X_b)
                loss = torch.nn.functional.mse_loss(pred, y_b)
                loss.backward()
                opt.step()

            model.eval()
            self.models.append(model)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and standard deviation across ensemble members."""
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(X_t).cpu().numpy())
        preds = np.array(preds)  # (n_models, n_samples)
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0)
        return mu, sigma
