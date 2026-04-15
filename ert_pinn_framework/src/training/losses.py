"""Loss containers and weighted composition for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossWeights:
    """Scalar weights for each loss component."""

    pde: float = 1.0
    dirichlet_bc: float = 1.0
    neumann_bc: float = 1.0
    data: float = 1.0
    regularization: float = 1e-4

    @classmethod
    def from_config(cls, config: dict) -> "LossWeights":
        """Build loss weights from training config section."""
        source = config.get("loss_weights", config)
        return cls(
            pde=float(source.get("pde", 1.0)),
            dirichlet_bc=float(source.get("dirichlet_bc", 1.0)),
            neumann_bc=float(source.get("neumann_bc", 1.0)),
            data=float(source.get("data", 1.0)),
            regularization=float(source.get("regularization", 1e-4)),
        )


class WeightedLossComposer:
    """Compose total objective from named components and configured weights."""

    def __init__(self, weights: LossWeights):
        self.weights = weights

    def total(self, losses: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Return weighted sum and a scalar log dictionary."""
        total = torch.tensor(0.0, device=self._device_from_losses(losses))

        weights_map = {
            "pde": self.weights.pde,
            "dirichlet_bc": self.weights.dirichlet_bc,
            "neumann_bc": self.weights.neumann_bc,
            "data": self.weights.data,
            "regularization": self.weights.regularization,
        }

        logs: dict[str, float] = {}
        for name, tensor_loss in losses.items():
            if name not in weights_map:
                continue
            weighted = float(weights_map[name]) * tensor_loss
            total = total + weighted
            logs[name] = float(tensor_loss.detach().cpu().item())

        logs["total"] = float(total.detach().cpu().item())
        return total, logs

    @staticmethod
    def _device_from_losses(losses: Mapping[str, Tensor]) -> torch.device:
        for value in losses.values():
            return value.device
        return torch.device("cpu")
