"""Repository-root wrapper for the forward synthetic dataset generator."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "ert_pinn_framework" / "scripts" / "forward" / "create_forward_synthetic_dataset.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")