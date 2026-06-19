"""Generate a robust synthetic benchmark suite for ERT inversion.

The suite trains one forward model, then builds multiple noisy and sparse
observation sets from the resulting predictions and runs inversion for each
scenario.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import load_project_config
from src.experiments.visual_simulation import apply_runtime_overrides, build_ert_array_observations
from src.main import run_minimal_inverse
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ERT benchmark with multiple noise and sparsity scenarios"
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
        default="exp_synthetic_benchmark",
        help="Name of the output experiment folder",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "quick", "standard"],
        default="quick",
        help="Simulation preset used for the base forward training run",
    )
    parser.add_argument(
        "--observation-counts",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Observation counts to sweep over",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03],
        help="Gaussian noise levels to sweep over",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of random replicates per observation/noise combination",
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
        help="Optional override for the candidate points used to build synthetic observations",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Robust synthetic ERT benchmark",
        help="Title for the generated visual diagnostics",
    )
    return parser.parse_args()


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _format_noise_std(noise_std: float) -> str:
    text = f"{noise_std:.4f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "neg").replace(".", "p")


def _scenario_name(observation_count: int, noise_std: float, replicate: int) -> str:
    return f"obs{observation_count:04d}_noise{_format_noise_std(noise_std)}_rep{replicate:02d}"


def _copy_warm_start_checkpoint(source_root: Path, target_root: Path) -> Path:
    source_checkpoint = source_root / "checkpoints" / "train_model.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {source_checkpoint}")

    target_checkpoint = target_root / "checkpoints" / "train_model.pt"
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, target_checkpoint)
    return target_checkpoint


def _aggregate_metrics(scenarios: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = ["rmse", "mae", "relative_l2"]
    collected: dict[str, list[float]] = {name: [] for name in metric_names}

    for scenario in scenarios:
        metrics = scenario.get("inversion_observation_fit", {})
        for name in metric_names:
            value = metrics.get(name)
            if value is not None:
                collected[name].append(float(value))

    aggregate: dict[str, dict[str, float]] = {}
    for name, values in collected.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        aggregate[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return aggregate


def main() -> None:
    args = parse_args()

    if args.replicates <= 0:
        raise ValueError("--replicates must be > 0")
    if not args.observation_counts:
        raise ValueError("--observation-counts must contain at least one value")
    if not args.noise_stds:
        raise ValueError("--noise-stds must contain at least one value")

    config = load_project_config(args.config_dir)
    apply_runtime_overrides(
        config,
        preset=args.preset,
        train_epochs=args.train_epochs,
        invert_epochs=args.invert_epochs,
        measurement_points=args.measurement_points,
    )

    project_root = Path(config["_meta"]["project_root"])
    output_root_rel = str(config["base"].get("paths", {}).get("output_root", "outputs"))
    output_root = project_root / output_root_rel / args.experiment_name
    base_training_root = output_root / "base_training"

    training_summary = run_minimal_inverse(config=config, output_root=base_training_root, mode="train")

    train_checkpoint = base_training_root / "checkpoints" / "train_model.pt"
    if not train_checkpoint.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {train_checkpoint}")

    base_seed = int(config["base"].get("project", {}).get("seed", 42))
    scenarios: list[dict[str, Any]] = []
    scenario_index = 0

    for observation_count in args.observation_counts:
        if observation_count <= 0:
            raise ValueError("Observation counts must be > 0")

        for noise_std in args.noise_stds:
            if noise_std < 0.0:
                raise ValueError("Noise standard deviations must be >= 0")

            for replicate in range(args.replicates):
                scenario_seed = base_seed + scenario_index * 100000 + replicate
                scenario_name = _scenario_name(observation_count, noise_std, replicate)
                scenario_root = output_root / "scenarios" / scenario_name
                scenario_root.mkdir(parents=True, exist_ok=True)

                observations_path = scenario_root / "synthetic_observations.csv"
                observations_info = build_ert_array_observations(
                    config=config,
                    checkpoint_path=train_checkpoint,
                    output_csv_path=observations_path,
                    observation_count=int(observation_count),
                    noise_std=float(noise_std),
                    seed=scenario_seed,
                )

                _copy_warm_start_checkpoint(base_training_root, scenario_root)

                scenario_config = copy.deepcopy(config)
                inverse_section = scenario_config["inverse"].get("inversion", scenario_config["inverse"])
                obs_cfg = inverse_section.setdefault("observations", {})
                obs_cfg["path"] = _safe_rel_path(observations_path, project_root)
                obs_cfg["delimiter"] = ","
                obs_cfg["skiprows"] = 1
                obs_cfg["point_columns"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                obs_cfg["value_column"] = 12

                inversion_summary = run_minimal_inverse(config=scenario_config, output_root=scenario_root, mode="invert")

                observation_fit = inversion_summary.get("observation_fit") or {}
                fit_metrics = observation_fit.get("metrics") or {}

                scenario_summary = {
                    "name": scenario_name,
                    "scenario_root": str(scenario_root),
                    "seed": int(scenario_seed),
                    "observation_count": int(observation_count),
                    "noise_std": float(noise_std),
                    "observations": observations_info,
                    "training_source": {
                        "root": str(base_training_root),
                        "predictions": str(train_checkpoint),
                    },
                    "inversion": inversion_summary,
                    "inversion_observation_fit": fit_metrics,
                }
                save_json(scenario_summary, scenario_root / "synthetic_scenario_summary.json")
                scenarios.append(scenario_summary)

            scenario_index += 1

    manifest = {
        "mode": "synthetic_benchmark_suite",
        "output_root": str(output_root),
        "base_training_root": str(base_training_root),
        "title": args.title,
        "recipe": {
            "preset": args.preset,
            "observation_counts": [int(value) for value in args.observation_counts],
            "noise_stds": [float(value) for value in args.noise_stds],
            "replicates": int(args.replicates),
            "train_epochs": args.train_epochs,
            "invert_epochs": args.invert_epochs,
            "measurement_points": args.measurement_points,
        },
        "training": training_summary,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "aggregate_metrics": _aggregate_metrics(scenarios),
    }
    save_json(manifest, output_root / "synthetic_benchmark_manifest.json")

    compact = {
        "mode": manifest["mode"],
        "output_root": manifest["output_root"],
        "base_training_root": manifest["base_training_root"],
        "scenario_count": manifest["scenario_count"],
        "manifest": str(output_root / "synthetic_benchmark_manifest.json"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()