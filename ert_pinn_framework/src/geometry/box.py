"""Box geometry implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .base import Geometry


FACE_SPECS: dict[str, tuple[int, float]] = {
    "x_min": (0, -1.0),
    "x_max": (0, 1.0),
    "y_min": (1, -1.0),
    "y_max": (1, 1.0),
    "z_min": (2, -1.0),
    "z_max": (2, 1.0),
}


class BoxGeometry(Geometry):
    """Cartesian box geometry."""

    def __init__(self, bounds: dict[str, tuple[float, float]]):
        super().__init__(bounds)

    def sample_interior(
        self,
        n_points: int,
        rng: np.random.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        mins = np.array([self.bounds["x"][0], self.bounds["y"][0], self.bounds["z"][0]], dtype=np.float64)
        maxs = np.array([self.bounds["x"][1], self.bounds["y"][1], self.bounds["z"][1]], dtype=np.float64)
        points = mins + rng.random((n_points, 3), dtype=np.float64) * (maxs - mins)
        return torch.tensor(points, device=device, dtype=dtype)

    def sample_boundary(
        self,
        face: str,
        n_points: int,
        rng: np.random.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if face not in FACE_SPECS:
            raise ValueError(f"Unknown face: {face}")

        points = self.sample_interior(n_points, rng, device, dtype)
        axis, sign = FACE_SPECS[face]

        fixed_value = self.bounds["xyz"[axis]][0] if sign < 0 else self.bounds["xyz"[axis]][1]
        points[:, axis] = torch.as_tensor(fixed_value, device=device, dtype=dtype)

        normals = torch.zeros((n_points, 3), device=device, dtype=dtype)
        normals[:, axis] = sign
        return points, normals
