"""Bradley-Terry pairwise preference model using an MLP utility function.

Learns a utility function u(x) such that P(x_i > x_j) = sigma(u(x_i) - u(x_j)),
where sigma is the logistic function. For BO, candidates are ranked by utility.

Reference:
    Bradley, R.A. & Terry, M.E. (1952). "Rank Analysis of Incomplete Block
    Designs: I. The Method of Paired Comparisons." Biometrika, 39(3/4), 324-345.
"""

import numpy as np
import torch

from .surrogate_mlp import SurrogateMLP


def _sample_pairs(indices, y, max_pairs, rng):
    """Sample pairwise comparisons from observations within a group."""
    n = len(indices)
    if n < 2:
        return [], []
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    pairs, seen = [], set()
    max_attempts = n_pairs * 20
    for _ in range(max_attempts):
        if len(pairs) >= n_pairs:
            break
        i, j = rng.choice(indices, size=2, replace=False)
        if y[i] == y[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        if rng.random() < 0.5:
            winner, loser = (i, j) if y[i] > y[j] else (j, i)
            pairs.append((winner, loser, 1))
        else:
            loser, winner = (i, j) if y[i] > y[j] else (j, i)
            pairs.append((loser, winner, 0))
    X1 = [(i, j) for i, j, _ in pairs]
    y1 = [lbl for _, _, lbl in pairs]
    return X1, y1


def build_pair_dataset(X, y, group_ids, mask, max_pairs=500, seed=42):
    """Build pairwise comparison dataset, sampling within groups."""
    rng = np.random.RandomState(seed)
    pairs, labels = [], []
    for gid in np.unique(group_ids[mask]):
        idx = np.where(mask & (group_ids == gid))[0]
        pair_idx, pair_labels = _sample_pairs(idx, y, max_pairs, rng)
        pairs.extend(pair_idx)
        labels.extend(pair_labels)
    return np.array(pairs), np.array(labels)


def train_bt_model(X, y, group_ids=None, epochs=20, batch_size=1024, seed=42):
    """Fit a Bradley-Terry model on pairwise comparisons.

    Parameters
    ----------
    X : array of shape (N, D), feature matrix.
    y : array of shape (N,), target values (used to derive preferences).
    group_ids : array of shape (N,), optional group/study IDs for stratified
        pair sampling. If None, all observations are treated as one group.
    epochs : Training epochs.
    batch_size : Batch size for pair training.
    seed : Random seed.

    Returns
    -------
    Fitted SurrogateMLP whose output is the learned utility u(x).
    """
    if group_ids is None:
        group_ids = np.zeros(len(y), dtype=int)
    mask = np.ones(len(y), dtype=bool)
    pairs, labels = build_pair_dataset(X, y, group_ids, mask, max_pairs=500, seed=seed)
    if len(pairs) == 0:
        # Fallback: not enough distinct pairs, return untrained model
        return SurrogateMLP(X.shape[1])

    model = SurrogateMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        perm = rng.permutation(len(pairs))
        for start in range(0, len(pairs), batch_size):
            batch = perm[start:start + batch_size]
            i_idx = pairs[batch, 0]
            j_idx = pairs[batch, 1]
            logits = model(X_t[i_idx]) - model(X_t[j_idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
