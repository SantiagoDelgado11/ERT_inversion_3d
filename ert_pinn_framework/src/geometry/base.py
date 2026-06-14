"""Base geometry module for the ERT PINN framework."""

from __future__ import annotations

import torch
from torch import Tensor
import numpy as np


class Geometry:
    """Abstract base class for all geometries."""

    def __init__(self, bounds: dict[str, tuple[float, float]]):
        self.bounds = bounds

    def sample_interior(
        self,
        n_points: int,
        rng: np.random.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        raise NotImplementedError

    def sample_boundary(
        self,
        face: str,
        n_points: int,
        rng: np.random.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
