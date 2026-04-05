"""GroupDRO surrogate for distributionally robust optimization across groups.

Minimizes the worst-group risk using exponentiated gradient ascent on group
weights. When groups correspond to experimental studies, this encourages
robust performance across diverse assay conditions.

Reference:
    Sagawa, S. et al. (2020). "Distributionally Robust Neural Networks for
    Group Shifts: On the Importance of Regularization for Worst-Case
    Generalization." ICLR 2020. arXiv:1911.08731.
"""

import numpy as np
import torch

from .surrogate_mlp import SurrogateMLP


def train_groupdro(X, y, group_ids, eta=0.01, epochs=200, lr=1e-3):
    """Fit an MLP using GroupDRO training objective.

    Parameters
    ----------
    X : array of shape (N, D), training features.
    y : array of shape (N,), training targets.
    group_ids : array of shape (N,), group/study IDs. Each unique ID defines
        a group whose loss is tracked independently.
    eta : float, exponentiated gradient step size for group weight updates.
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
    q = np.ones(len(unique_groups)) / len(unique_groups)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        losses = []
        for gid in unique_groups:
            idx = group_indices[gid]
            pred = model(X_t[idx])
            loss = torch.mean((pred - y_t[idx]) ** 2)
            losses.append(loss)

        losses_t = torch.stack(losses)
        with torch.no_grad():
            q = q * np.exp(eta * losses_t.cpu().numpy())
            q = q / q.sum()
        q_t = torch.tensor(q, dtype=losses_t.dtype)

        loss = torch.sum(q_t.detach() * losses_t)
        loss.backward()
        optimizer.step()

    return model
