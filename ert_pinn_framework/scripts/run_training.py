"""CLI entry point for forward PINN training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERT PINN forward training")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "configs"),
        help="Path to configs directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        config_dir=args.config_dir,
        override_mode="train",
        override_name=args.experiment_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
