"""Electrodes module."""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import Tensor


@dataclasses.dataclass
class ElectrodeSet:
    """Represents a set of electrodes."""
    positions: Tensor
    ids: Optional[list[str]] = None
    metadata: Optional[dict] = None

    def __len__(self) -> int:
        return self.positions.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        return self.positions[idx]


def build_circular_electrodes(
    count: int,
    radius: float,
    z_value: float,
    device: torch.device,
    dtype: torch.dtype
) -> ElectrodeSet:
    """Builds a circular ring of electrodes."""
    angles = torch.linspace(0.0, 2.0 * torch.pi, steps=count + 1, device=device, dtype=dtype)[:-1]
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    z = torch.full((count,), z_value, device=device, dtype=dtype)
    positions = torch.stack([x, y, z], dim=1)
    
    ids = [str(i) for i in range(count)]
    return ElectrodeSet(positions=positions, ids=ids)
