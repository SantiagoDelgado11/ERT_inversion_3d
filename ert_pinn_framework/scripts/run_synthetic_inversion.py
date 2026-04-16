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

from src.engine.pipelines import run_inversion_pipeline, run_training_pipeline
from src.evaluation.metrics import summary_metrics
from src.experiments.runner import load_project_config
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
        "--data-loss-weight",
        type=float,
        default=1.0,
        help="Weight for inversion data term",
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


def _experiment_section(experiment_cfg: dict) -> dict:
    return experiment_cfg.get("experiment", experiment_cfg)


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
        training["batch_size"] = 2048
        training["checkpoint_every"] = 50
        inversion["checkpoint_every"] = 50
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

    inversion.setdefault("loss_weights", {})["data"] = float(args.data_loss_weight)
    if args.data_loss_weight > 0.0:
        inversion["jointly_optimize_potential"] = True


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


def _optional_array(npz_data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in npz_data.files:
        return None
    value = npz_data[key]
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1 and value.item() is None:
        return None
    return np.asarray(value)


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


def _plot_potential_comparison(
    points: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    y_true_1d = _flatten_column(y_true)
    y_pred_1d = _flatten_column(y_pred)
    error = y_pred_1d - y_true_1d

    min_common = float(np.min([y_true_1d.min(), y_pred_1d.min()]))
    max_common = float(np.max([y_true_1d.max(), y_pred_1d.max()]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    sc0 = axes[0].scatter(
        points[:, 0],
        points[:, 1],
        c=y_true_1d,
        s=12,
        cmap="viridis",
        vmin=min_common,
        vmax=max_common,
    )
    axes[0].set_title("Potencial verdadero")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    sc1 = axes[1].scatter(
        points[:, 0],
        points[:, 1],
        c=y_pred_1d,
        s=12,
        cmap="viridis",
        vmin=min_common,
        vmax=max_common,
    )
    axes[1].set_title("Potencial predicho")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    sc2 = axes[2].scatter(
        points[:, 0],
        points[:, 1],
        c=error,
        s=12,
        cmap="coolwarm",
    )
    axes[2].set_title("Error (pred - true)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.colorbar(sc0, ax=axes[0], label="potential")
    fig.colorbar(sc1, ax=axes[1], label="potential")
    fig.colorbar(sc2, ax=axes[2], label="error")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return summary_metrics(y_true_1d, y_pred_1d)


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

    train_points = _optional_array(data, "train_points")
    train_true = _optional_array(data, "train_potential_true")
    train_pred = _optional_array(data, "train_potential_pred")
    train_sigma = _optional_array(data, "train_conductivity_pred")

    if train_points is not None and train_true is not None and train_pred is not None:
        potential_fig = plot_dir / "train_potential_comparison_xy.png"
        potential_metrics = _plot_potential_comparison(
            points=train_points,
            y_true=train_true,
            y_pred=train_pred,
            output_path=potential_fig,
        )
        results["figures"].append(str(potential_fig))
        results["metrics"]["train_potential"] = potential_metrics

    val_points = _optional_array(data, "val_points")
    val_true = _optional_array(data, "val_potential_true")
    val_pred = _optional_array(data, "val_potential_pred")
    if val_points is not None and val_true is not None and val_pred is not None:
        val_fig = plot_dir / "val_potential_comparison_xy.png"
        val_metrics = _plot_potential_comparison(
            points=val_points,
            y_true=val_true,
            y_pred=val_pred,
            output_path=val_fig,
        )
        results["figures"].append(str(val_fig))
        results["metrics"]["val_potential"] = val_metrics

    if train_points is not None and train_sigma is not None:
        sigma_1d = _flatten_column(train_sigma)
        sigma_fig = plot_dir / "train_conductivity_xy.png"
        _plot_scatter_panel(
            points=train_points,
            values=sigma_1d,
            title="Conductividad invertida (train points)",
            output_path=sigma_fig,
            cmap="plasma",
        )
        results["figures"].append(str(sigma_fig))
        results["metrics"]["train_conductivity"] = {
            "mean": float(np.mean(sigma_1d)),
            "std": float(np.std(sigma_1d)),
            "min": float(np.min(sigma_1d)),
            "max": float(np.max(sigma_1d)),
        }

    generic_points = _optional_array(data, "points")
    generic_sigma = _optional_array(data, "conductivity")
    if generic_points is not None and generic_sigma is not None:
        sigma_1d = _flatten_column(generic_sigma)
        sigma_fig = plot_dir / "conductivity_xy.png"
        _plot_scatter_panel(
            points=generic_points,
            values=sigma_1d,
            title="Conductividad invertida",
            output_path=sigma_fig,
            cmap="plasma",
        )
        results["figures"].append(str(sigma_fig))
        results["metrics"]["conductivity"] = {
            "mean": float(np.mean(sigma_1d)),
            "std": float(np.std(sigma_1d)),
            "min": float(np.min(sigma_1d)),
            "max": float(np.max(sigma_1d)),
        }

    metrics_path = plot_dir / "metrics.json"
    save_json(results["metrics"], metrics_path)
    results["metrics_path"] = str(metrics_path)
    return results


def main() -> None:
    args = parse_args()

    config = load_project_config(args.config_dir)

    exp_section = _experiment_section(config["experiment"])
    exp_section["name"] = args.experiment_name
    exp_section["save_predictions"] = True
    exp_section["save_checkpoints"] = True

    _apply_runtime_overrides(config, args)

    project_root = Path(config["_meta"]["project_root"])
    output_root_rel = str(config["base"].get("paths", {}).get("output_root", "outputs"))
    output_root = project_root / output_root_rel / args.experiment_name

    training_summary = run_training_pipeline(config, output_root=output_root, force_final_checkpoint=True)

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

    inversion_summary = run_inversion_pipeline(config, output_root=output_root)
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
