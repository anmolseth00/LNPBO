"""V-REx surrogate for out-of-distribution generalization via risk extrapolation.

Penalizes the variance of per-group losses to encourage invariant predictive
performance across experimental studies/environments.

Reference:
    Krueger, D. et al. (2021). "Out-of-Distribution Generalization via Risk
    Extrapolation (REx)." ICML 2021. arXiv:2003.00688.
"""

import numpy as np
import torch

from .surrogate_mlp import SurrogateMLP


def train_vrex(X, y, group_ids, lambda_rex=1.0, epochs=200, lr=1e-3):
    """Fit an MLP using V-REx training objective.

    The loss is: mean(per_group_losses) + lambda_rex * var(per_group_losses).

    Parameters
    ----------
    X : array of shape (N, D), training features.
    y : array of shape (N,), training targets.
    group_ids : array of shape (N,), group/study IDs.
    lambda_rex : float, penalty weight on loss variance across groups.
    epochs : Training epochs.
    lr : Learning rate.

    Returns
    -------
    Fitted SurrogateMLP.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = SurrogateMLP(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    unique_groups = np.unique(group_ids)
    group_indices = {gid: np.where(group_ids == gid)[0] for gid in unique_groups}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for gid in unique_groups:
            idx = group_indices[gid]
            pred = model(X_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            losses.append(loss)

        if not losses:
            break

        losses_t = torch.stack(losses)
        loss = losses_t.mean() + lambda_rex * losses_t.var(unbiased=False)
        loss.backward()
        optimizer.step()

    return model
