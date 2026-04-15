"""Optimizer and scheduler factories."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def build_optimizer(parameters, config: dict) -> Optimizer:
    """Create optimizer from configuration dictionary."""
    source = config.get("optimizer", config)
    name = str(source.get("name", "adam")).lower()
    lr = float(source.get("lr", 1e-3))
    weight_decay = float(source.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    if name == "sgd":
        momentum = float(source.get("momentum", 0.9))
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if name == "lbfgs":
        max_iter = int(source.get("max_iter", 20))
        return torch.optim.LBFGS(parameters, lr=lr, max_iter=max_iter)

    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: Optimizer, config: dict):
    """Create scheduler from configuration dictionary or return None."""
    source = config.get("scheduler", config)
    enabled = bool(source.get("enabled", False))
    if not enabled:
        return None

    name = str(source.get("name", "step_lr")).lower()

    if name == "step_lr":
        step_size = int(source.get("step_size", 100))
        gamma = float(source.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        t_max = int(source.get("t_max", 500))
        eta_min = float(source.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name == "reduce_on_plateau":
        factor = float(source.get("factor", 0.5))
        patience = int(source.get("patience", 20))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

    raise ValueError(f"Unsupported scheduler: {name}")
