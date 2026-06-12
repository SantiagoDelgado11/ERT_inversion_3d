"""Electric potential network for the ERT PINN."""

from __future__ import annotations

from torch import Tensor, nn

from .mlp import MLP


class PotentialNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int, activation: str):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
