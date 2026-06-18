"""CLI entry point for conductivity inversion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.cli_overrides import add_pinn_runtime_args, build_pinn_runtime_overrides
from src.experiments.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ERT PINN inversion")
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
    parser.add_argument(
        "--observations-path",
        "--input-data",
        dest="observations_path",
        type=str,
        default=None,
        help="Path to a real observation CSV or NPZ file",
    )
    parser.add_argument(
        "--observations-format",
        type=str,
        default=None,
        choices=["auto", "csv", "npz"],
        help="Override the observation file format",
    )
    parser.add_argument(
        "--observations-delimiter",
        type=str,
        default=None,
        help="Delimiter used by the observation file",
    )
    parser.add_argument(
        "--observations-skiprows",
        type=int,
        default=None,
        help="Number of initial rows to skip in the observation file",
    )
    parser.add_argument(
        "--observations-has-header",
        action="store_true",
        help="Treat the observation table as header-based",
    )
    parser.add_argument(
        "--observations-point-columns",
        type=int,
        nargs="+",
        default=None,
        help="Positional columns for the x,y,z coordinates (3 or 12 columns)",
    )
    parser.add_argument(
        "--observations-value-column",
        type=int,
        default=None,
        help="Positional column for the observed value",
    )
    parser.add_argument(
        "--observations-point-names",
        type=str,
        nargs=3,
        default=None,
        help="Named columns for the x, y and z coordinates",
    )
    parser.add_argument(
        "--observations-value-name",
        type=str,
        default=None,
        help="Named column for the observed value",
    )
    parser.add_argument(
        "--observations-drop-invalid-rows",
        action="store_true",
        help="Skip rows with non-numeric or incomplete observation values",
    )
    add_pinn_runtime_args(parser, include_warm_start=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observation_overrides = {
        "path": args.observations_path,
        "format": args.observations_format,
        "delimiter": args.observations_delimiter,
        "skiprows": args.observations_skiprows,
        "has_header": True if args.observations_has_header else None,
        "point_columns": args.observations_point_columns,
        "value_column": args.observations_value_column,
        "point_column_names": args.observations_point_names,
        "value_column_name": args.observations_value_name,
        "drop_invalid_rows": True if args.observations_drop_invalid_rows else None,
    }
    summary = run_experiment(
        config_dir=args.config_dir,
        override_mode="invert",
        override_name=args.experiment_name,
        observation_overrides=observation_overrides,
        runtime_overrides=build_pinn_runtime_overrides(args, mode="invert"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
