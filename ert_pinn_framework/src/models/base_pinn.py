"""Abstract base class for physics-informed neural networks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ..physics.operators import gradient, laplacian


class BasePINN(nn.Module, ABC):
    """Base class that provides common PINN helper methods."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Compute network output from input coordinates."""

    def potential(self, x: Tensor) -> Tensor:
        """Return scalar potential prediction."""
        return self.forward(x)

    def grad_potential(self, x: Tensor) -> Tensor:
        """Compute first derivatives of the potential with respect to inputs."""
        x_req = self._with_grad(x)
        u = self.potential(x_req)
        return gradient(u, x_req)

    def laplacian_potential(self, x: Tensor) -> Tensor:
        """Compute Laplacian of the predicted potential."""
        x_req = self._with_grad(x)
        u = self.potential(x_req)
        return laplacian(u, x_req)

    @property
    def device(self) -> torch.device:
        """Return device where module parameters currently live."""
        return next(self.parameters()).device

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _with_grad(x: Tensor) -> Tensor:
        if x.requires_grad:
            return x
        return x.clone().detach().requires_grad_(True)
