"""Boundary-condition helpers and loss functions."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .operators import ensure_requires_grad, gradient


ConductivityLike = float | Tensor | Callable[[Tensor], Tensor]


def dirichlet_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean-squared error for Dirichlet boundary values."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have same shape")
    return torch.mean((prediction - target) ** 2)


def box_face_normals(
    points: Tensor,
    bounds: dict[str, tuple[float, float]],
    atol: float = 1e-6,
) -> Tensor:
    """Infer outward face normals for points on a box boundary."""
    normals = torch.zeros_like(points)

    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    z_min, z_max = bounds["z"]

    normals[torch.isclose(points[:, 0], torch.as_tensor(x_min, device=points.device, dtype=points.dtype), atol=atol), 0] = -1.0
    normals[torch.isclose(points[:, 0], torch.as_tensor(x_max, device=points.device, dtype=points.dtype), atol=atol), 0] = 1.0
    normals[torch.isclose(points[:, 1], torch.as_tensor(y_min, device=points.device, dtype=points.dtype), atol=atol), 1] = -1.0
    normals[torch.isclose(points[:, 1], torch.as_tensor(y_max, device=points.device, dtype=points.dtype), atol=atol), 1] = 1.0
    normals[torch.isclose(points[:, 2], torch.as_tensor(z_min, device=points.device, dtype=points.dtype), atol=atol), 2] = -1.0
    normals[torch.isclose(points[:, 2], torch.as_tensor(z_max, device=points.device, dtype=points.dtype), atol=atol), 2] = 1.0

    missing = torch.isclose(torch.linalg.norm(normals, dim=1), torch.tensor(0.0, device=points.device, dtype=points.dtype))
    if torch.any(missing):
        raise ValueError("Some boundary points do not match any box face within atol")

    return normals


def _evaluate_conductivity(conductivity: ConductivityLike, points: Tensor) -> Tensor:
    if callable(conductivity):
        sigma = conductivity(points)
    elif isinstance(conductivity, Tensor):
        sigma = conductivity
    else:
        sigma = torch.full((points.shape[0], 1), float(conductivity), device=points.device, dtype=points.dtype)

    if not isinstance(sigma, Tensor):
        sigma = torch.as_tensor(sigma, device=points.device, dtype=points.dtype)
    else:
        sigma = sigma.to(device=points.device, dtype=points.dtype)

    if sigma.ndim == 0:
        return sigma.view(1, 1).expand(points.shape[0], 1)
    if sigma.ndim == 1:
        if sigma.numel() == 1:
            return sigma.view(1, 1).expand(points.shape[0], 1)
        if sigma.shape[0] == points.shape[0]:
            return sigma.unsqueeze(-1)
        raise ValueError("1D conductivity must have length 1 or N")
    if sigma.ndim == 2:
        if sigma.shape == (1, 1):
            return sigma.expand(points.shape[0], 1)
        if sigma.shape == (points.shape[0], 1):
            return sigma
        raise ValueError("2D conductivity must have shape (1, 1) or (N, 1)")
    raise ValueError("conductivity must be scalar-like or broadcastable to (N, 1)")


def _normal_flux(model, points: Tensor, normals: Tensor, conductivity: ConductivityLike) -> Tensor:
    x = ensure_requires_grad(points)
    u = model.potential(x)
    grad_u = gradient(u, x)

    if normals.shape != grad_u.shape:
        raise ValueError("normals must have shape (N, D) matching point gradients")

    normals = normals.to(device=x.device, dtype=x.dtype)
    sigma = _evaluate_conductivity(conductivity, x)
    return torch.sum(sigma * grad_u * normals, dim=1, keepdim=True)


def neumann_loss(
    model,
    points: Tensor,
    normals: Tensor,
    target_flux: Tensor,
    conductivity: ConductivityLike = 1.0,
) -> Tensor:
    """Mean-squared error for Neumann flux boundary condition."""
    x = ensure_requires_grad(points)
    if target_flux.ndim == 1:
        target_flux = target_flux.unsqueeze(-1)
    if target_flux.shape != (x.shape[0], 1):
        raise ValueError("target_flux must have shape (N, 1)")
    target_flux = target_flux.to(device=x.device, dtype=x.dtype)

    normal_flux = _normal_flux(model=model, points=x, normals=normals, conductivity=conductivity)
    return torch.mean((normal_flux - target_flux) ** 2)


def flux_conservation_loss(
    model,
    source_points: Tensor,
    source_normals: Tensor,
    sink_points: Tensor,
    sink_normals: Tensor,
    current: float,
    surface_area: float,
    conductivity: ConductivityLike = 1.0,
) -> Tensor:
    """cPINN flux conservation loss on source/sink control surfaces."""
    if surface_area <= 0.0:
        raise ValueError("surface_area must be > 0")

    source_flux = _normal_flux(
        model=model,
        points=source_points,
        normals=source_normals,
        conductivity=conductivity,
    )
    sink_flux = _normal_flux(
        model=model,
        points=sink_points,
        normals=sink_normals,
        conductivity=conductivity,
    )

    target = torch.as_tensor(float(current / surface_area), device=source_flux.device, dtype=source_flux.dtype)
    source_mean = torch.mean(source_flux)
    sink_mean = torch.mean(sink_flux)

    return (source_mean - target) ** 2 + (sink_mean + target) ** 2
