"""Run or visualize a synthetic ERT inversion simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.visual_simulation import create_visualization_suite, run_visual_inversion_simulation


def _resolve_output_root(path_value: str | None, experiment_name: str) -> Path:
    if path_value is None:
        return PROJECT_ROOT / "outputs" / experiment_name

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic ERT inversion scenario and generate visual diagnostics"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(PROJECT_ROOT / "configs"),
        help="Path to the configs directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="exp_visual_simulation",
        help="Output experiment folder name",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "quick", "standard"],
        default="tiny",
        help="Simulation size preset",
    )
    parser.add_argument(
        "--observation-count",
        type=int,
        default=256,
        help="Synthetic potential observations sampled from forward predictions; <=0 keeps all",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to synthetic potential observations",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Optional override for forward training epochs",
    )
    parser.add_argument(
        "--invert-epochs",
        type=int,
        default=None,
        help="Optional override for inversion epochs",
    )
    parser.add_argument(
        "--measurement-points",
        type=int,
        default=None,
        help="Optional override for prediction points used in visualizations",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Simulacion visual de inversion ERT",
        help="Title used in the main visualization",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only create visualizations from an existing output folder",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Existing output folder for --skip-run, relative to project root if not absolute",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_run:
        output_root = _resolve_output_root(args.output_root, args.experiment_name)
        summary = {
            "mode": "visualize_existing_inversion",
            "output_root": str(output_root),
            "visualization": create_visualization_suite(output_root, title=args.title),
        }
    else:
        summary = run_visual_inversion_simulation(
            config_dir=args.config_dir,
            experiment_name=args.experiment_name,
            preset=args.preset,
            observation_count=args.observation_count,
            noise_std=args.noise_std,
            train_epochs=args.train_epochs,
            invert_epochs=args.invert_epochs,
            measurement_points=args.measurement_points,
            title=args.title,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
