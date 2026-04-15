"""Sampling strategies for collocation and boundary points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import FACE_NAMES, BoxDomain3D


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for one collocation sampling pass."""

    interior_points: int
    boundary_points_per_face: int
    random_seed: int = 42

    @classmethod
    def from_config(cls, config: dict) -> "SamplerConfig":
        """Build sampler config from loaded YAML dictionary."""
        sampling = config.get("sampling", {})
        return cls(
            interior_points=int(sampling.get("interior_points_per_epoch", 0)),
            boundary_points_per_face=int(sampling.get("boundary_points_per_face_per_epoch", 0)),
            random_seed=int(sampling.get("random_seed", 42)),
        )

    def validate(self) -> None:
        if self.interior_points <= 0:
            raise ValueError("interior_points must be > 0")
        if self.boundary_points_per_face <= 0:
            raise ValueError("boundary_points_per_face must be > 0")


@dataclass(frozen=True)
class CollocationBatch:
    """Container for interior and boundary collocation points."""

    interior: np.ndarray
    boundaries: dict[str, np.ndarray]

    @property
    def all_boundary_points(self) -> np.ndarray:
        """Stack all boundary points in face order."""
        return np.vstack([self.boundaries[face] for face in FACE_NAMES])


class DomainSampler:
    """Sampler object that yields deterministic collocation batches."""

    def __init__(self, config: SamplerConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def sample(self, domain: BoxDomain3D) -> CollocationBatch:
        """Sample one collocation batch from the domain."""
        interior = domain.sample_uniform(self.config.interior_points, self.rng)
        boundaries = {
            face: domain.sample_face(face, self.config.boundary_points_per_face, self.rng)
            for face in FACE_NAMES
        }
        return CollocationBatch(interior=interior, boundaries=boundaries)
