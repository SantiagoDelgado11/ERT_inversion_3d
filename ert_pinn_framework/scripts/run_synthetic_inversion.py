"""Run synthetic ERT inversion end-to-end and save visual results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import load_project_config
from src.main import run_minimal_inverse
from src.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward training, build synthetic observations, invert conductivity, and plot results"
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
        default="exp_synthetic",
        help="Name of the output experiment folder",
    )
    parser.add_argument(
        "--observation-count",
        type=int,
        default=512,
        help="Number of synthetic observations sampled from training predictions (<=0 keeps all)",
    )
    parser.add_argument(
        "--measurement-points",
        type=int,
        default=None,
        help="Number of forward prediction points generated before synthetic observation sampling",
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
        "--quick",
        action="store_true",
        help="Use a faster setup for quick validation",
    )
    return parser.parse_args()


def _inversion_section(inverse_cfg: dict) -> dict:
    return inverse_cfg.get("inversion", inverse_cfg)


def _training_section(training_cfg: dict) -> dict:
    return training_cfg.get("training", training_cfg)


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _apply_runtime_overrides(config: dict, args: argparse.Namespace) -> None:
    training = _training_section(config["training"])
    inversion = _inversion_section(config["inverse"])
    sampling = config["data"].setdefault("sampling", {})

    if args.quick:
        training["epochs"] = 300
        inversion["epochs"] = 200
        sampling["interior_points_per_epoch"] = 6000
        sampling["boundary_points_per_face_per_epoch"] = 600
        training["log_every"] = 10
        inversion["log_every"] = 10

    if args.train_epochs is not None:
        training["epochs"] = int(args.train_epochs)
    if args.invert_epochs is not None:
        inversion["epochs"] = int(args.invert_epochs)
    if args.measurement_points is not None:
        measurement_points = int(args.measurement_points)
        if measurement_points <= 0:
            raise ValueError("--measurement-points must be > 0")
        sampling["measurement_points"] = measurement_points


def _build_synthetic_observations(
    training_predictions_path: Path,
    output_csv_path: Path,
    observation_count: int,
    noise_std: float,
    seed: int,
) -> dict:
    data = np.load(training_predictions_path)
    points = np.asarray(data["points"], dtype=np.float64)
    potential = np.asarray(data["potential"], dtype=np.float64).reshape(-1, 1)

    if points.shape[0] != potential.shape[0]:
        raise ValueError("Mismatch between number of points and potential predictions")

    rng = np.random.default_rng(seed)
    if 0 < observation_count < points.shape[0]:
        idx = rng.choice(points.shape[0], size=observation_count, replace=False)
        points = points[idx]
        potential = potential[idx]

    if noise_std > 0.0:
        potential = potential + rng.normal(loc=0.0, scale=noise_std, size=potential.shape)

    table = np.hstack([points, potential])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv_path,
        table,
        delimiter=",",
        header="x,y,z,potential",
        comments="# ",
    )

    return {
        "path": str(output_csv_path),
        "count": int(points.shape[0]),
        "noise_std": float(noise_std),
    }


def _flatten_column(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def _plot_scatter_panel(
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sc = ax.scatter(points[:, 0], points[:, 1], c=values, s=14, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, label="value")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_plots(output_root: Path) -> dict:
    npz_path = output_root / "inversion_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Inversion predictions not found: {npz_path}")

    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    results: dict[str, object] = {
        "plot_dir": str(plot_dir),
        "figures": [],
        "metrics": {},
    }

    if "points" not in data.files or "conductivity" not in data.files:
        raise KeyError(f"Unsupported NPZ keys in {npz_path}. Expected points and conductivity, got {data.files}")

    points = np.asarray(data["points"], dtype=np.float64)
    sigma = _flatten_column(np.asarray(data["conductivity"], dtype=np.float64))
    sigma_fig = plot_dir / "conductivity_xy.png"
    _plot_scatter_panel(
        points=points,
        values=sigma,
        title="Conductividad invertida",
        output_path=sigma_fig,
        cmap="plasma",
    )
    results["figures"].append(str(sigma_fig))
    results["metrics"]["conductivity"] = {
        "mean": float(np.mean(sigma)),
        "std": float(np.std(sigma)),
        "min": float(np.min(sigma)),
        "max": float(np.max(sigma)),
    }

    metrics_path = plot_dir / "metrics.json"
    save_json(results["metrics"], metrics_path)
    results["metrics_path"] = str(metrics_path)
    return results


def main() -> None:
    args = parse_args()

    config = load_project_config(args.config_dir)

    _apply_runtime_overrides(config, args)

    project_root = Path(config["_meta"]["project_root"])
    output_root_rel = str(config["base"].get("paths", {}).get("output_root", "outputs"))
    output_root = project_root / output_root_rel / args.experiment_name

    training_summary = run_minimal_inverse(config=config, output_root=output_root, mode="train")

    predictions_path = output_root / "training_predictions.npz"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Training predictions not found: {predictions_path}")

    project_seed = int(config["base"].get("project", {}).get("seed", 42))
    observations_path = output_root / "synthetic_observations.csv"
    observations_info = _build_synthetic_observations(
        training_predictions_path=predictions_path,
        output_csv_path=observations_path,
        observation_count=int(args.observation_count),
        noise_std=float(args.noise_std),
        seed=project_seed,
    )

    inv_section = _inversion_section(config["inverse"])
    obs_cfg = inv_section.setdefault("observations", {})
    obs_cfg["path"] = _safe_rel_path(observations_path, project_root)
    obs_cfg["skiprows"] = 1

    inversion_summary = run_minimal_inverse(config=config, output_root=output_root, mode="invert")
    plots_summary = _make_plots(output_root)

    full_summary = {
        "mode": "synthetic_train_then_invert",
        "output_root": str(output_root),
        "training": training_summary,
        "observations": observations_info,
        "inversion": inversion_summary,
        "plots": plots_summary,
    }
    save_json(full_summary, output_root / "synthetic_run_summary.json")

    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
