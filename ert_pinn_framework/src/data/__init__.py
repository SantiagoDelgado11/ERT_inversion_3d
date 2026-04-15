"""Data domain and sampling utilities."""

from .domain import BoxDomain3D
from .sampler import CollocationBatch, DomainSampler, SamplerConfig

__all__ = [
    "BoxDomain3D",
    "SamplerConfig",
    "CollocationBatch",
    "DomainSampler",
]
