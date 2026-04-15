"""Forward solver based on PINN residual minimization."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from ..physics.pde import conductivity_pde_residual


class ForwardSolver:
    """Light wrapper to evaluate potential and PDE residual with a PINN model."""

    def __init__(self, model, conductivity: float | Tensor | Callable[[Tensor], Tensor] = 1.0, source=None):
        self.model = model
        self.conductivity = conductivity
        self.source = source

    def potential(self, points: Tensor) -> Tensor:
        """Predict potential field at points."""
        return self.model.potential(points)

    def residual(self, points: Tensor) -> Tensor:
        """Evaluate PDE residual at collocation points."""
        return conductivity_pde_residual(
            model=self.model,
            points=points,
            conductivity=self.conductivity,
            source=self.source,
        )

    def pde_loss(self, points: Tensor) -> Tensor:
        """Mean-squared PDE residual."""
        residual = self.residual(points)
        return torch.mean(residual**2)
