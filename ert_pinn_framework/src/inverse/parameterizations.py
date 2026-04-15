"""Parameterizations that enforce positive conductivity fields."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def _positive_transform(raw: Tensor, mode: str) -> Tensor:
    key = mode.lower()
    if key == "exp":
        return torch.exp(raw)
    if key == "softplus":
        return torch.nn.functional.softplus(raw)
    raise ValueError(f"Unsupported positivity transform: {mode}")


class BaseConductivityParameterization(nn.Module, ABC):
    """Abstract base for conductivity parameter fields."""

    @abstractmethod
    def forward(self, points: Tensor) -> Tensor:
        """Evaluate conductivity at points with output shape (N, 1)."""


class LogConductivityParameterization(BaseConductivityParameterization):
    """Global conductivity parameterized as sigma = exp(theta)."""

    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        init_sigma_value = float(init_sigma)
        if init_sigma_value <= 0.0:
            raise ValueError("init_sigma must be > 0 for log-conductivity")
        init_log_sigma = torch.log(torch.tensor(init_sigma_value))
        self.log_sigma = nn.Parameter(init_log_sigma.view(1, 1))

    def forward(self, points: Tensor) -> Tensor:
        sigma = torch.exp(self.log_sigma)
        return sigma.expand(points.shape[0], 1)


class SoftplusConductivityParameterization(BaseConductivityParameterization):
    """Global conductivity parameterized as sigma = softplus(theta)."""

    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        init_sigma_value = max(float(init_sigma), 1e-8)
        inv_softplus = torch.log(torch.expm1(torch.tensor(init_sigma_value)))
        self.raw_sigma = nn.Parameter(inv_softplus.view(1, 1))

    def forward(self, points: Tensor) -> Tensor:
        sigma = torch.nn.functional.softplus(self.raw_sigma)
        return sigma.expand(points.shape[0], 1)


class MLPConductivityParameterization(BaseConductivityParameterization):
    """Spatial conductivity field represented with a small MLP."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        positivity: str = "exp",
    ):
        super().__init__()
        self.positivity = positivity

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, points: Tensor) -> Tensor:
        raw = self.network(points)
        return _positive_transform(raw, self.positivity)


def build_conductivity_parameterization(config: dict) -> BaseConductivityParameterization:
    """Factory for conductivity parameterizations from config."""
    source = config.get("inversion", config)
    kind = str(source.get("conductivity_parameterization", "log_conductivity")).lower()
    positivity = str(source.get("positivity_enforcement", "exp")).lower()

    bounds = source.get("bounds", {})
    sigma_min = float(bounds.get("conductivity_min", 0.001))
    sigma_max = float(bounds.get("conductivity_max", 10.0))
    if sigma_min <= 0.0 or sigma_max <= 0.0:
        raise ValueError("conductivity bounds must be strictly positive")
    if sigma_min >= sigma_max:
        raise ValueError("conductivity_min must be smaller than conductivity_max")
    init_sigma = 0.5 * (sigma_min + sigma_max)

    if kind == "log_conductivity":
        return LogConductivityParameterization(init_sigma=init_sigma)

    if kind == "softplus":
        return SoftplusConductivityParameterization(init_sigma=init_sigma)

    if kind in {"mlp", "mlp_conductivity", "field_mlp"}:
        return MLPConductivityParameterization(
            input_dim=int(source.get("input_dim", 3)),
            hidden_dim=int(source.get("hidden_dim", 64)),
            num_hidden_layers=int(source.get("num_hidden_layers", 3)),
            positivity=positivity,
        )

    raise ValueError(f"Unsupported conductivity_parameterization: {kind}")
