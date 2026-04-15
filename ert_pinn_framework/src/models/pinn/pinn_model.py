"""Multilayer perceptron PINN with optional Fourier features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..base_pinn import BasePINN


def _get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class FourierFeatures(nn.Module):
    """Frozen random Fourier feature mapping."""

    def __init__(self, input_dim: int, feature_dim: int, sigma: float = 1.0):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        weight = torch.randn(input_dim, feature_dim) * float(sigma)
        self.register_buffer("weight", weight, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        projection = 2.0 * math.pi * x @ self.weight
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class PINNModel(BasePINN):
    """Fully connected PINN used for potential field approximation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_hidden_layers: int = 6,
        activation: str = "tanh",
        use_fourier_features: bool = False,
        fourier_features_dim: int = 0,
        fourier_sigma: float = 1.0,
        weight_init: str = "xavier_uniform",
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)

        self.use_fourier_features = bool(use_fourier_features)
        self.fourier: FourierFeatures | None = None

        first_dim = input_dim
        if self.use_fourier_features:
            self.fourier = FourierFeatures(
                input_dim=input_dim,
                feature_dim=fourier_features_dim,
                sigma=fourier_sigma,
            )
            first_dim = 2 * fourier_features_dim

        act = _get_activation(activation)
        layers: list[nn.Module] = [nn.Linear(first_dim, hidden_dim), act]

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(_get_activation(activation))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

        self._initialize_weights(weight_init)

    @classmethod
    def from_config(cls, config: dict) -> "PINNModel":
        """Build model from loaded model configuration."""
        source = config.get("model", config)
        return cls(
            input_dim=int(source.get("input_dim", 3)),
            output_dim=int(source.get("output_dim", 1)),
            hidden_dim=int(source.get("hidden_dim", 128)),
            num_hidden_layers=int(source.get("num_hidden_layers", 6)),
            activation=str(source.get("activation", "tanh")),
            use_fourier_features=bool(source.get("use_fourier_features", False)),
            fourier_features_dim=int(source.get("fourier_features_dim", 0)),
            fourier_sigma=float(source.get("fourier_sigma", 1.0)),
            weight_init=str(source.get("weight_init", "xavier_uniform")),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_feat = self.fourier(x) if self.fourier is not None else x
        return self.network(x_feat)

    def _initialize_weights(self, strategy: str) -> None:
        key = strategy.lower()
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue

            if key == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif key == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif key == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unsupported weight_init: {strategy}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)
