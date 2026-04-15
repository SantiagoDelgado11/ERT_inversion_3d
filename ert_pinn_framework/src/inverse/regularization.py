"""Regularization terms used during inversion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..physics.operators import gradient


def l2_parameter_regularization(module: nn.Module) -> Tensor:
    """L2 penalty over all trainable parameters in a module."""
    penalties = [torch.sum(p**2) for p in module.parameters() if p.requires_grad]
    if not penalties:
        first_param = next(module.parameters(), None)
        if first_param is None:
            return torch.tensor(0.0)
        return torch.zeros((), device=first_param.device, dtype=first_param.dtype)
    return torch.stack(penalties).sum()


def total_variation_regularization(field_values: Tensor, points: Tensor, eps: float = 1e-8) -> Tensor:
    """Isotropic total-variation-like penalty based on gradient magnitude."""
    grad_field = gradient(field_values, points)
    grad_norm = torch.sqrt(torch.sum(grad_field**2, dim=1) + eps)
    return torch.mean(grad_norm)
