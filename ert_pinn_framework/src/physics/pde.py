"""PDE residual definitions for the ERT forward problem."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .operators import divergence, ensure_requires_grad, gradient


ConductivityLike = float | Tensor | Callable[[Tensor], Tensor]
SourceLike = None | float | Tensor | Callable[[Tensor], Tensor]


def _evaluate_field(field: ConductivityLike | SourceLike, points: Tensor, default: float) -> Tensor:
    n_points = points.shape[0]

    if field is None:
        value = torch.full((n_points, 1), float(default), device=points.device, dtype=points.dtype)
        return value

    if callable(field):
        value = field(points)
    elif isinstance(field, Tensor):
        value = field
    else:
        value = torch.as_tensor(field, device=points.device, dtype=points.dtype)

    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, device=points.device, dtype=points.dtype)
    else:
        value = value.to(device=points.device, dtype=points.dtype)

    if value.ndim == 0:
        return value.view(1, 1).expand(n_points, 1)

    if value.ndim == 1:
        if value.numel() == 1:
            return value.view(1, 1).expand(n_points, 1)
        if value.shape[0] == n_points:
            return value.unsqueeze(-1)
        raise ValueError("1D field must have length 1 or N")

    if value.ndim == 2:
        if value.shape == (1, 1):
            return value.expand(n_points, 1)
        if value.shape == (n_points, 1):
            return value
        raise ValueError("2D field must have shape (1, 1) or (N, 1)")

    raise ValueError("Field must be scalar-like or broadcastable to shape (N, 1)")


def conductivity_pde_residual(
    model,
    points: Tensor,
    conductivity: ConductivityLike = 1.0,
    source: SourceLike = None,
) -> Tensor:
    """Compute residual of -div(sigma * grad(u)) = source."""
    x = ensure_requires_grad(points)
    u = model.potential(x)
    grad_u = gradient(u, x)

    sigma = _evaluate_field(conductivity, x, default=1.0)
    flux = sigma * grad_u

    div_flux = divergence(flux, x)
    src = _evaluate_field(source, x, default=0.0)
    return -div_flux - src
