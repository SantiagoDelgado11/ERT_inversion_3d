"""Differential operators implemented with torch autograd."""

from __future__ import annotations

import torch
from torch import Tensor


def ensure_requires_grad(x: Tensor) -> Tensor:
    """Ensure tensor tracks gradients for autograd-based derivatives."""
    if x.requires_grad:
        return x
    return x.clone().detach().requires_grad_(True)


def gradient(scalar_field: Tensor, inputs: Tensor) -> Tensor:
    """Compute gradient of a scalar field with respect to input coordinates."""
    if scalar_field.ndim != 2 or scalar_field.shape[1] != 1:
        raise ValueError("scalar_field must have shape (N, 1)")

    grads = torch.autograd.grad(
        outputs=scalar_field,
        inputs=inputs,
        grad_outputs=torch.ones_like(scalar_field),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return grads


def divergence(vector_field: Tensor, inputs: Tensor) -> Tensor:
    """Compute divergence of vector field with shape (N, D)."""
    if vector_field.ndim != 2:
        raise ValueError("vector_field must have shape (N, D)")
    if inputs.ndim != 2 or inputs.shape[1] != vector_field.shape[1]:
        raise ValueError("inputs must have shape (N, D) and match vector_field dimensions")

    dim = vector_field.shape[1]
    div_terms = []
    for axis in range(dim):
        component = vector_field[:, axis : axis + 1]
        grad_component = torch.autograd.grad(
            outputs=component,
            inputs=inputs,
            grad_outputs=torch.ones_like(component),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        div_terms.append(grad_component[:, axis : axis + 1])

    return torch.stack(div_terms, dim=0).sum(dim=0)


def laplacian(scalar_field: Tensor, inputs: Tensor) -> Tensor:
    """Compute Laplacian of scalar field."""
    grad_u = gradient(scalar_field, inputs)
    return divergence(grad_u, inputs)
