"""Electrical conductivity network for the ERT PINN."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .mlp import MLP


class ConductivityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
        sigma_floor: float = 1e-6,
    ):
        super().__init__()
        self.sigma_floor = float(sigma_floor)
        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(self.model(x)) + self.sigma_floor
