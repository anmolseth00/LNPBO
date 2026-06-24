"""Shared MLP base class for neural surrogate models.

Used by SNGP, Laplace, Bradley-Terry, GroupDRO, and V-REx surrogates.
"""

import torch
import torch.nn.functional as F


class SurrogateMLP(torch.nn.Module):
    """Three-layer MLP (256 -> 128 -> 1) for regression or utility estimation."""

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)
