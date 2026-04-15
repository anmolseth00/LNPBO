"""Meta-training for the exact FSBO deep-kernel surrogate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch

from .encoder import FSBOFeatureExtractor, to_tensor
from .kernel import DeepKernelExactGP

logger = logging.getLogger("lnpbo")


@dataclass
class FSBOMetaState:
    hidden_dims: tuple[int, ...]
    base_kernel: str
    num_mixtures: int
    feature_extractor_state: dict[str, torch.Tensor]
    mean_state: dict[str, torch.Tensor]
    covar_state: dict[str, torch.Tensor]
    likelihood_state: dict[str, torch.Tensor]
    y_global_bounds: tuple[float, float]
    meta_losses: list[float]


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def sample_task_scaling_bounds(
    y_min: float,
    y_max: float,
    rng: np.random.RandomState,
) -> tuple[float, float]:
    lower, upper = rng.uniform(y_min, y_max, size=2)
    if lower > upper:
        lower, upper = upper, lower
    if upper - lower < 1e-8:
        upper = lower + 1e-8
    return float(lower), float(upper)


def augment_task_labels(y: np.ndarray, lower: float, upper: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return (y - lower) / max(upper - lower, 1e-8)


def meta_train_fsbo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    study_ids_train: np.ndarray,
    train_study_ids: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (128, 128),
    base_kernel: str = "rbf",
    num_mixtures: int = 4,
    batch_size: int = 50,
    batches_per_task: int = 1,
    n_iterations: int = 300,
    lr_kernel: float = 1e-3,
    lr_feature_extractor: float = 1e-3,
    seed: int = 42,
) -> FSBOMetaState:
    """Meta-train FSBO by Algorithm 1 / Eq. 8-11."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    study_ids_train = np.asarray(study_ids_train)
    if isinstance(train_study_ids, set):
        train_study_ids = sorted(train_study_ids)
    else:
        train_study_ids = list(train_study_ids)
    train_study_ids = np.asarray(train_study_ids)

    task_indices = {
        sid: np.where(study_ids_train == sid)[0]
        for sid in sorted(train_study_ids)
        if np.any(study_ids_train == sid)
    }
    if not task_indices:
        raise ValueError("No source tasks available for FSBO meta-training.")

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    first_task = next(iter(task_indices))
    init_idx = task_indices[first_task]
    init_batch = rng.choice(init_idx, size=min(batch_size, len(init_idx)), replace=False)
    init_x = to_tensor(X_train[init_batch])
    init_y = torch.tensor(y_train[init_batch], dtype=torch.float64)

    feature_extractor = FSBOFeatureExtractor(init_x.shape[-1], hidden_dims).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = DeepKernelExactGP(
        init_x,
        init_y,
        likelihood,
        feature_extractor=feature_extractor,
        base_kernel=base_kernel,
        num_mixtures=num_mixtures,
    ).double()
    model.initialize_kernel_from_batch(init_x, init_y)

    optimizer = torch.optim.Adam(
        [
            {"params": list(model.feature_extractor.parameters()), "lr": lr_feature_extractor},
            {
                "params": (
                    list(model.mean_module.parameters())
                    + list(model.covar_module.parameters())
                    + list(model.likelihood.parameters())
                ),
                "lr": lr_kernel,
            },
        ]
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    global_y_min = float(np.min(y_train))
    global_y_max = float(np.max(y_train))
    source_ids = list(task_indices.keys())
    losses: list[float] = []

    for step in range(n_iterations):
        task_id = source_ids[int(rng.randint(len(source_ids)))]
        lower, upper = sample_task_scaling_bounds(global_y_min, global_y_max, rng)

        for _ in range(max(1, batches_per_task)):
            idx = task_indices[task_id]
            batch_idx = rng.choice(idx, size=min(batch_size, len(idx)), replace=False)
            batch_x = to_tensor(X_train[batch_idx])
            batch_y = torch.tensor(
                augment_task_labels(y_train[batch_idx], lower, upper),
                dtype=torch.float64,
            )

            model.train()
            likelihood.train()
            model.set_train_data(batch_x, batch_y, strict=False)

            optimizer.zero_grad()
            loss = -mll(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        if (step + 1) % 50 == 0:
            logger.info("  FSBO meta-step %d/%d: nll=%.4f", step + 1, n_iterations, np.mean(losses[-10:]))

    model.eval()
    likelihood.eval()
    return FSBOMetaState(
        hidden_dims=hidden_dims,
        base_kernel=base_kernel,
        num_mixtures=num_mixtures,
        feature_extractor_state=clone_state_dict(model.feature_extractor),
        mean_state=clone_state_dict(model.mean_module),
        covar_state=clone_state_dict(model.covar_module),
        likelihood_state=clone_state_dict(likelihood),
        y_global_bounds=(global_y_min, global_y_max),
        meta_losses=losses,
    )
