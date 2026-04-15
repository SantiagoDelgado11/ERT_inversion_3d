"""Simple function registry for experiment modes."""

from __future__ import annotations

from typing import Callable


_EXPERIMENTS: dict[str, Callable] = {}


def register_experiment(name: str, fn: Callable) -> None:
    """Register an experiment pipeline function."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Experiment name cannot be empty")
    _EXPERIMENTS[key] = fn


def get_experiment(name: str) -> Callable:
    """Retrieve experiment function by name."""
    register_default_experiments()
    key = name.strip().lower()
    if key not in _EXPERIMENTS:
        options = ", ".join(sorted(_EXPERIMENTS.keys()))
        raise KeyError(f"Unknown experiment mode '{name}'. Available: {options}")
    return _EXPERIMENTS[key]


def available_experiments() -> list[str]:
    """Return sorted list of available experiment names."""
    register_default_experiments()
    return sorted(_EXPERIMENTS.keys())


def register_default_experiments() -> None:
    """Register built-in experiment pipelines once."""
    from ..engine.pipelines import (
        run_inversion_pipeline,
        run_train_then_invert_pipeline,
        run_training_pipeline,
    )

    defaults = {
        "train": run_training_pipeline,
        "training": run_training_pipeline,
        "invert": run_inversion_pipeline,
        "inversion": run_inversion_pipeline,
        "train_then_invert": run_train_then_invert_pipeline,
    }
    for name, fn in defaults.items():
        _EXPERIMENTS.setdefault(name, fn)
