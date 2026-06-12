"""Shared MLP backbone used by the PINN component networks."""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str,
    ):
        super().__init__()
        act_name = activation.lower()
        if act_name == "tanh":
            act = nn.Tanh
        elif act_name == "relu":
            act = nn.ReLU
        elif act_name == "silu":
            act = nn.SiLU
        elif act_name == "gelu":
            act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act()]
        for _ in range(max(0, hidden_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
