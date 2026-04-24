"""Minimal compatibility wrappers for experiment modes.

The full inverse PINN implementation now lives in src/main.py.
"""

from __future__ import annotations

from pathlib import Path

from ..main import run_minimal_inverse
from ..utils.io import save_json


def run_training_pipeline(config: dict, output_root: Path, force_final_checkpoint: bool = False) -> dict:
    del force_final_checkpoint
    return run_minimal_inverse(config=config, output_root=Path(output_root), mode="train")


def run_inversion_pipeline(config: dict, output_root: Path) -> dict:
    return run_minimal_inverse(config=config, output_root=Path(output_root), mode="invert")


def run_train_then_invert_pipeline(config: dict, output_root: Path) -> dict:
    training_summary = run_minimal_inverse(config=config, output_root=Path(output_root), mode="train")
    inversion_summary = run_minimal_inverse(config=config, output_root=Path(output_root), mode="invert")

    summary = {
        "mode": "train_then_invert",
        "training": training_summary,
        "inversion": inversion_summary,
    }
    save_json(summary, Path(output_root) / "experiment_summary.json")
    return summary
