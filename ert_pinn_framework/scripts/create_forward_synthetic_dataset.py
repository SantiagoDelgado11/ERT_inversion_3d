"""Create a robust synthetic dataset from the 3D PINN forward model.

The script can either reuse an existing ``training_predictions.npz`` file or
train the forward model first. It then builds one or more noisy dataset
variants, each split into train/validation/test CSV files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import load_project_config
from src.experiments.visual_simulation import apply_runtime_overrides
from src.main import run_minimal_inverse
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a synthetic ERT dataset from the PINN forward model"
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
        default="exp_forward_synthetic_dataset",
        help="Name of the output experiment folder",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "quick", "standard"],
        default="standard",
        help="Forward-model preset used when the predictions are generated in this run",
    )
    parser.add_argument(
        "--training-predictions-path",
        type=str,
        default=None,
        help="Reuse an existing training_predictions.npz file instead of running forward training",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Optional override for forward training epochs when generating predictions in this run",
    )
    parser.add_argument(
        "--measurement-points",
        type=int,
        default=8192,
        help="Number of forward prediction points used as the candidate pool",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=-1,
        help="Number of points to keep from the candidate pool; <=0 keeps all points",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        help="Train/val/test split ratios",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.03],
        help="Noise levels to generate as robust dataset variants",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=2,
        help="Number of random replicates per noise level",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Synthetic forward PINN dataset",
        help="Human-readable title stored in the manifest",
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


def _array_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_csv(path: Path, points: np.ndarray, potential: np.ndarray) -> None:
    table = np.hstack([points, potential.reshape(-1, 1)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="x,y,z,potential",
        comments="# ",
        fmt="%.10e",
    )


def _split_counts(total: int, split_ratios: list[float]) -> tuple[int, int, int]:
    if total <= 0:
        raise ValueError("total must be > 0")
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain exactly 3 values")

    fractions = np.asarray(split_ratios, dtype=np.float64)
    if np.any(fractions < 0.0):
        raise ValueError("split ratios must be non-negative")
    if float(np.sum(fractions)) <= 0.0:
        raise ValueError("split ratios must sum to a positive value")

    fractions = fractions / np.sum(fractions)
    raw = fractions * float(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total - int(np.sum(counts)))

    fractions_order = np.argsort((raw - counts))[::-1]
    for index in range(remainder):
        counts[fractions_order[index % 3]] += 1

    if total >= 3:
        for index in range(3):
            if counts[index] > 0:
                continue
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                continue
            counts[donor] -= 1
            counts[index] += 1

    if int(np.sum(counts)) != total:
        raise RuntimeError("Split count computation failed to preserve total sample count")

    return int(counts[0]), int(counts[1]), int(counts[2])


def _split_arrays(
    points: np.ndarray,
    potential: np.ndarray,
    *,
    split_ratios: list[float],
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    total = int(points.shape[0])
    train_count, val_count, test_count = _split_counts(total, split_ratios)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(total)

    train_idx = permutation[:train_count]
    val_idx = permutation[train_count : train_count + val_count]
    test_idx = permutation[train_count + val_count : train_count + val_count + test_count]

    return {
        "train": {"points": points[train_idx], "potential": potential[train_idx]},
        "val": {"points": points[val_idx], "potential": potential[val_idx]},
        "test": {"points": points[test_idx], "potential": potential[test_idx]},
    }


def _load_candidate_pool(predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Training predictions not found: {predictions_path}")

    data = np.load(predictions_path)
    if "points" not in data.files or "potential" not in data.files:
        raise KeyError(f"Expected points and potential in {predictions_path}; got {data.files}")

    points = np.asarray(data["points"], dtype=np.float64)
    potential = np.asarray(data["potential"], dtype=np.float64).reshape(-1)
    conductivity = np.asarray(data["conductivity"], dtype=np.float64).reshape(-1) if "conductivity" in data.files else None

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point array with shape (N, 3) in {predictions_path}; got {points.shape}")
    if points.shape[0] != potential.shape[0]:
        raise ValueError("Point and potential arrays must have the same number of rows")
    if conductivity is not None and conductivity.shape[0] != points.shape[0]:
        raise ValueError("Point and conductivity arrays must have the same number of rows")

    return points, potential, conductivity


def main() -> None:
    args = parse_args()

    if args.replicates <= 0:
        raise ValueError("--replicates must be > 0")
    if not args.noise_stds:
        raise ValueError("--noise-stds must contain at least one value")

    config = load_project_config(args.config_dir)
    apply_runtime_overrides(
        config,
        preset=args.preset,
        train_epochs=args.train_epochs,
        measurement_points=args.measurement_points,
    )

    project_root = Path(config["_meta"]["project_root"])
    output_root_rel = str(config["base"].get("paths", {}).get("output_root", "outputs"))
    output_root = project_root / output_root_rel / args.experiment_name
    dataset_root = output_root / "dataset"
    forward_root = output_root / "forward_model"
    dataset_root.mkdir(parents=True, exist_ok=True)

    training_summary: dict[str, Any] | None = None
    if args.training_predictions_path is None:
        training_summary = run_minimal_inverse(config=config, output_root=forward_root, mode="train")
        predictions_path = forward_root / "training_predictions.npz"
    else:
        predictions_path = Path(args.training_predictions_path)
        if not predictions_path.is_absolute():
            predictions_path = project_root / predictions_path

    points, clean_potential, conductivity = _load_candidate_pool(predictions_path)

    candidate_size = int(points.shape[0]) if args.dataset_size <= 0 else min(int(args.dataset_size), int(points.shape[0]))
    base_seed = int(config["base"].get("project", {}).get("seed", 42))
    manifest_variants: list[dict[str, Any]] = []

    for noise_index, noise_std in enumerate(args.noise_stds):
        if noise_std < 0.0:
            raise ValueError("Noise standard deviations must be >= 0")

        for replicate in range(args.replicates):
            variant_seed = base_seed + noise_index * 100000 + replicate
            variant_name = f"noise_{_format_noise_std(noise_std)}_rep{replicate:02d}"
            variant_root = dataset_root / "variants" / variant_name
            variant_root.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(variant_seed)
            if candidate_size < points.shape[0]:
                subset_idx = rng.choice(points.shape[0], size=candidate_size, replace=False)
                subset_points = points[subset_idx]
                subset_potential = clean_potential[subset_idx]
                subset_conductivity = conductivity[subset_idx] if conductivity is not None else None
            else:
                subset_points = points.copy()
                subset_potential = clean_potential.copy()
                subset_conductivity = conductivity.copy() if conductivity is not None else None

            if noise_std > 0.0:
                subset_potential = subset_potential + rng.normal(loc=0.0, scale=float(noise_std), size=subset_potential.shape)

            split = _split_arrays(
                subset_points,
                subset_potential,
                split_ratios=[float(v) for v in args.split_ratios],
                seed=variant_seed,
            )

            full_csv_path = variant_root / "full.csv"
            train_csv_path = variant_root / "train.csv"
            val_csv_path = variant_root / "val.csv"
            test_csv_path = variant_root / "test.csv"

            _write_csv(full_csv_path, subset_points, subset_potential)
            _write_csv(train_csv_path, split["train"]["points"], split["train"]["potential"])
            _write_csv(val_csv_path, split["val"]["points"], split["val"]["potential"])
            _write_csv(test_csv_path, split["test"]["points"], split["test"]["potential"])

            candidate_npz_path = variant_root / "candidate_pool.npz"
            if subset_conductivity is None:
                np.savez_compressed(candidate_npz_path, points=subset_points, potential=subset_potential)
            else:
                np.savez_compressed(
                    candidate_npz_path,
                    points=subset_points,
                    potential=subset_potential,
                    conductivity=subset_conductivity,
                )

            variant_summary = {
                "name": variant_name,
                "root": str(variant_root),
                "seed": int(variant_seed),
                "noise_std": float(noise_std),
                "candidate_count": int(subset_points.shape[0]),
                "split_counts": {
                    "train": int(split["train"]["points"].shape[0]),
                    "val": int(split["val"]["points"].shape[0]),
                    "test": int(split["test"]["points"].shape[0]),
                },
                "files": {
                    "full": _safe_rel_path(full_csv_path, project_root),
                    "train": _safe_rel_path(train_csv_path, project_root),
                    "val": _safe_rel_path(val_csv_path, project_root),
                    "test": _safe_rel_path(test_csv_path, project_root),
                    "candidate_pool": _safe_rel_path(candidate_npz_path, project_root),
                },
                "stats": {
                    "points": {
                        "x": [float(np.min(subset_points[:, 0])), float(np.max(subset_points[:, 0]))],
                        "y": [float(np.min(subset_points[:, 1])), float(np.max(subset_points[:, 1]))],
                        "z": [float(np.min(subset_points[:, 2])), float(np.max(subset_points[:, 2]))],
                    },
                    "potential": _array_stats(subset_potential),
                },
            }
            if subset_conductivity is not None:
                variant_summary["stats"]["conductivity"] = _array_stats(subset_conductivity)

            save_json(variant_summary, variant_root / "dataset_variant_summary.json")
            manifest_variants.append(variant_summary)

    recommended_variant = manifest_variants[0] if manifest_variants else None
    manifest = {
        "mode": "forward_synthetic_dataset",
        "title": args.title,
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "candidate_pool_path": _safe_rel_path(predictions_path, project_root),
        "forward_root": str(forward_root),
        "forward_training": training_summary,
        "dataset_size": int(candidate_size),
        "split_ratios": [float(v) for v in args.split_ratios],
        "noise_stds": [float(v) for v in args.noise_stds],
        "replicates": int(args.replicates),
        "variant_count": len(manifest_variants),
        "variants": manifest_variants,
        "recommended_train_variant": None if recommended_variant is None else recommended_variant["files"]["train"],
    }
    save_json(manifest, dataset_root / "dataset_manifest.json")

    compact = {
        "mode": manifest["mode"],
        "dataset_root": manifest["dataset_root"],
        "manifest": _safe_rel_path(dataset_root / "dataset_manifest.json", project_root),
        "recommended_train_variant": manifest["recommended_train_variant"],
        "variant_count": manifest["variant_count"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()