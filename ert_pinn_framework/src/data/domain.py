"""Geometric domain definitions for 3D ERT sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


@dataclass(frozen=True)
class BoxDomain3D:
    """Axis-aligned 3D box domain with utility sampling methods."""

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        self._validate_bounds(self.x_bounds, "x")
        self._validate_bounds(self.y_bounds, "y")
        self._validate_bounds(self.z_bounds, "z")

    @staticmethod
    def _validate_bounds(bounds: tuple[float, float], axis: str) -> None:
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"Invalid {axis}-bounds: expected min < max, got {bounds}")

    @classmethod
    def from_config(cls, config: dict) -> "BoxDomain3D":
        """Construct domain from a config dictionary."""
        source = config.get("domain", config)
        bounds = source["bounds"]
        return cls(
            x_bounds=tuple(bounds["x"]),
            y_bounds=tuple(bounds["y"]),
            z_bounds=tuple(bounds["z"]),
        )

    @property
    def mins(self) -> np.ndarray:
        return np.array([self.x_bounds[0], self.y_bounds[0], self.z_bounds[0]], dtype=np.float64)

    @property
    def maxs(self) -> np.ndarray:
        return np.array([self.x_bounds[1], self.y_bounds[1], self.z_bounds[1]], dtype=np.float64)

    @property
    def lengths(self) -> np.ndarray:
        return self.maxs - self.mins

    def contains(self, points: np.ndarray, atol: float = 0.0) -> np.ndarray:
        """Return boolean mask indicating whether points are inside the box."""
        points = self._as_points(points)
        lower = points >= (self.mins - atol)
        upper = points <= (self.maxs + atol)
        return np.logical_and(lower, upper).all(axis=1)

    def sample_uniform(self, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sample uniformly inside the volume."""
        if n_points <= 0:
            raise ValueError("n_points must be > 0")
        u = rng.random((n_points, 3), dtype=np.float64)
        return self.mins + u * self.lengths

    def sample_face(self, face: str, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sample points uniformly on a named box face."""
        if face not in FACE_NAMES:
            raise ValueError(f"Unknown face '{face}'. Valid faces: {FACE_NAMES}")
        if n_points <= 0:
            raise ValueError("n_points must be > 0")

        points = self.sample_uniform(n_points, rng)
        if face == "x_min":
            points[:, 0] = self.x_bounds[0]
        elif face == "x_max":
            points[:, 0] = self.x_bounds[1]
        elif face == "y_min":
            points[:, 1] = self.y_bounds[0]
        elif face == "y_max":
            points[:, 1] = self.y_bounds[1]
        elif face == "z_min":
            points[:, 2] = self.z_bounds[0]
        elif face == "z_max":
            points[:, 2] = self.z_bounds[1]
        return points

    def boundary_masks(self, points: np.ndarray, atol: float = 1e-8) -> dict[str, np.ndarray]:
        """Create masks for points close to each box face."""
        points = self._as_points(points)
        return {
            "x_min": np.isclose(points[:, 0], self.x_bounds[0], atol=atol),
            "x_max": np.isclose(points[:, 0], self.x_bounds[1], atol=atol),
            "y_min": np.isclose(points[:, 1], self.y_bounds[0], atol=atol),
            "y_max": np.isclose(points[:, 1], self.y_bounds[1], atol=atol),
            "z_min": np.isclose(points[:, 2], self.z_bounds[0], atol=atol),
            "z_max": np.isclose(points[:, 2], self.z_bounds[1], atol=atol),
        }

    @staticmethod
    def _as_points(points: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return arr
