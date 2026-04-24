"""PDE residual definitions for the ERT forward problem."""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from .operators import divergence, ensure_requires_grad, gradient


ConductivityLike = float | Tensor | Callable[[Tensor], Tensor]
SourceLike = None | float | Tensor | Callable[[Tensor], Tensor]


def _as_center(center: Tensor | list[float] | tuple[float, ...], points: Tensor) -> Tensor:
    center_tensor = center if isinstance(center, Tensor) else torch.as_tensor(center, device=points.device, dtype=points.dtype)
    if not isinstance(center_tensor, Tensor):
        center_tensor = torch.as_tensor(center_tensor, device=points.device, dtype=points.dtype)
    center_tensor = center_tensor.to(device=points.device, dtype=points.dtype).reshape(-1)
    if center_tensor.shape[0] != points.shape[1]:
        raise ValueError("center must have one coordinate per spatial dimension")
    return center_tensor


def gaussian_smoothed_delta(
    points: Tensor,
    center: Tensor | list[float] | tuple[float, ...],
    epsilon: float,
) -> Tensor:
    """Evaluate a normalized Gaussian approximation of the Dirac distribution."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive")

    center_tensor = _as_center(center, points)
    diff = points - center_tensor.unsqueeze(0)
    squared_radius = torch.sum(diff * diff, dim=1, keepdim=True)

    dim = points.shape[1]
    sigma = float(epsilon)
    normalization = (2.0 * math.pi) ** (0.5 * dim) * (sigma**dim)
    exponent = -0.5 * squared_radius / (sigma * sigma)
    return torch.exp(exponent) / normalization


def gaussian_dipole_source(
    points: Tensor,
    source_center: Tensor | list[float] | tuple[float, ...],
    sink_center: Tensor | list[float] | tuple[float, ...],
    current: float,
    epsilon: float,
) -> Tensor:
    """Compute I*delta_eps(x-x_A) - I*delta_eps(x-x_B) using Gaussian kernels."""
    source_term = gaussian_smoothed_delta(points=points, center=source_center, epsilon=epsilon)
    sink_term = gaussian_smoothed_delta(points=points, center=sink_center, epsilon=epsilon)
    current_tensor = torch.as_tensor(float(current), device=points.device, dtype=points.dtype)
    return current_tensor * (source_term - sink_term)


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


def _div_sigma_grad_u_product_rule(model, x: Tensor, conductivity: ConductivityLike) -> Tensor:
    """Compute div(sigma*grad(u)) as sigma*laplace(u) + grad(sigma)·grad(u)."""
    u = model.potential(x)
    grad_u = gradient(u, x)
    sigma = _evaluate_field(conductivity, x, default=1.0)

    lap_u = divergence(grad_u, x)

    if sigma.requires_grad:
        grad_sigma = torch.autograd.grad(
            outputs=sigma,
            inputs=x,
            grad_outputs=torch.ones_like(sigma),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_sigma is None:
            grad_sigma = torch.zeros_like(x)
    else:
        grad_sigma = torch.zeros_like(x)

    return sigma * lap_u + torch.sum(grad_sigma * grad_u, dim=1, keepdim=True)


def conductivity_pde_residual(
    model,
    points: Tensor,
    conductivity: ConductivityLike = 1.0,
    source: SourceLike = None,
) -> Tensor:
    """Compute residual of -div(sigma * grad(u)) = source."""
    x = ensure_requires_grad(points)
    div_flux = _div_sigma_grad_u_product_rule(model=model, x=x, conductivity=conductivity)
    src = _evaluate_field(source, x, default=0.0)
    return -div_flux - src
