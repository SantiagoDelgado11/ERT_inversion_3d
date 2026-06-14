"""Feature builders for PINN."""

from __future__ import annotations

import torch
from torch import Tensor


def build_potential_features(
    points: Tensor,
    source_center: Tensor,
    sink_center: Tensor,
    current: float | Tensor
) -> Tensor:
    """
    Build 22-dimensional features for PotentialNet.
    
    Args:
        points: (N, 3) tensor of spatial coordinates
        source_center: (3,) or (1, 3) tensor of source electrode position
        sink_center: (3,) or (1, 3) tensor of sink electrode position
        current: scalar or (1,) tensor of injection current
        
    Returns:
        Tensor of shape (N, 22) containing:
        - point (3)
        - source (3)
        - sink (3)
        - current (1)
        - r_source (3)
        - r_sink (3)
        - dist_source (1)
        - dist_sink (1)
        - dipole_vector (3)
        - dipole_length (1)
    """
    N = points.shape[0]
    device = points.device
    dtype = points.dtype

    src = source_center.view(1, 3).expand(N, 3)
    snk = sink_center.view(1, 3).expand(N, 3)
    
    if isinstance(current, float):
        curr = torch.full((N, 1), current, device=device, dtype=dtype)
    else:
        curr = current.view(1, 1).expand(N, 1)

    r_src = points - src
    r_snk = points - snk
    
    dist_src = torch.linalg.vector_norm(r_src, dim=1, keepdim=True)
    dist_snk = torch.linalg.vector_norm(r_snk, dim=1, keepdim=True)
    
    dipole_vec = src - snk
    dipole_len = torch.linalg.vector_norm(dipole_vec, dim=1, keepdim=True)

    features = torch.cat([
        points,         # 3
        src,            # 3
        snk,            # 3
        curr,           # 1
        r_src,          # 3
        r_snk,          # 3
        dist_src,       # 1
        dist_snk,       # 1
        dipole_vec,     # 3
        dipole_len      # 1
    ], dim=1)           # Total: 22

    return features
