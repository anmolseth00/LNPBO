"""Feature preprocessing and shared deep-kernel encoders for exact FSBO."""

from __future__ import annotations

import numpy as np
import torch


def normalize_features(
    X: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Min-max normalize features to ``[0, 1]`` and return the bounds used."""
    X = np.asarray(X, dtype=np.float64)
    if bounds is None:
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        rng = x_max - x_min
        rng[rng < 1e-8] = 1.0
        bounds = (x_min, rng)
    x_min, rng = bounds
    return (X - x_min) / rng, bounds


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(X), dtype=torch.float64)


class IdentityFeatures(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FSBOFeatureExtractor(torch.nn.Module):
    """Two-layer MLP feature extractor used by the paper's deep kernel."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer.")
        layers: list[torch.nn.Module] = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims) - 1:
                layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        self.net = torch.nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
