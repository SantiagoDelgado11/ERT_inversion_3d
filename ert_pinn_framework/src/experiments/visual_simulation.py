"""Synthetic visual simulation workflow for ERT inversion outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..main import run_minimal_inverse
from ..utils.io import save_json
from .runner import load_project_config


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key, config)
    if not isinstance(section, dict):
        raise TypeError(f"Expected mapping for section '{key}'")
    return section


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _array_stats(values: np.ndarray) -> dict[str, float]:
    arr = _flatten(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = _flatten(y_true)
    pred = _flatten(y_pred)
    err = pred - true
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "relative_l2": float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12)),
    }


def _downsample(points: np.ndarray, values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, values

    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx], values[idx]


def _read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def _apply_visual_preset(config: dict[str, Any], preset: str) -> None:
    model_cfg = _section(config["model"], "model")
    training_cfg = _section(config["training"], "training")
    inverse_cfg = _section(config["inverse"], "inversion")
    sampling_cfg = config["data"].setdefault("sampling", {})

    normalized = preset.strip().lower()
    if normalized == "tiny":
        model_cfg["hidden_dim"] = 32
        model_cfg["num_hidden_layers"] = 2
        inverse_cfg["hidden_dim"] = 32
        inverse_cfg["num_hidden_layers"] = 2
        sampling_cfg["interior_points_per_epoch"] = 256
        sampling_cfg["boundary_points_per_face_per_epoch"] = 64
        sampling_cfg["measurement_points"] = 1500
        training_cfg["epochs"] = 5
        training_cfg["log_every"] = 1
        inverse_cfg["epochs"] = 5
        inverse_cfg["log_every"] = 1
        return

    if normalized == "quick":
        model_cfg["hidden_dim"] = 64
        model_cfg["num_hidden_layers"] = 3
        inverse_cfg["hidden_dim"] = 64
        inverse_cfg["num_hidden_layers"] = 3
        sampling_cfg["interior_points_per_epoch"] = 4000
        sampling_cfg["boundary_points_per_face_per_epoch"] = 400
        sampling_cfg["measurement_points"] = 4000
        training_cfg["epochs"] = 300
        training_cfg["log_every"] = 10
        inverse_cfg["epochs"] = 200
        inverse_cfg["log_every"] = 10
        return

    if normalized == "standard":
        return

    raise ValueError("preset must be one of: tiny, quick, standard")


def apply_runtime_overrides(
    config: dict[str, Any],
    *,
    preset: str,
    train_epochs: int | None = None,
    invert_epochs: int | None = None,
    measurement_points: int | None = None,
) -> None:
    """Apply a named simulation preset and optional CLI-level overrides."""
    _apply_visual_preset(config, preset)

    training_cfg = _section(config["training"], "training")
    inverse_cfg = _section(config["inverse"], "inversion")
    sampling_cfg = config["data"].setdefault("sampling", {})

    if train_epochs is not None:
        training_cfg["epochs"] = int(train_epochs)
    if invert_epochs is not None:
        inverse_cfg["epochs"] = int(invert_epochs)
    if measurement_points is not None:
        value = int(measurement_points)
        if value <= 0:
            raise ValueError("measurement_points must be > 0")
        sampling_cfg["measurement_points"] = value


def build_synthetic_observations(
    training_predictions_path: Path,
    output_csv_path: Path,
    *,
    observation_count: int,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    """Sample synthetic potential observations from forward PINN predictions."""
    data = np.load(training_predictions_path)
    points = np.asarray(data["points"], dtype=np.float64)
    potential = np.asarray(data["potential"], dtype=np.float64).reshape(-1, 1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Training prediction points must have shape (N, 3)")
    if points.shape[0] != potential.shape[0]:
        raise ValueError("Potential predictions must match the number of points")

    rng = np.random.default_rng(seed)
    if 0 < observation_count < points.shape[0]:
        idx = rng.choice(points.shape[0], size=observation_count, replace=False)
        points = points[idx]
        potential = potential[idx]

    if noise_std > 0.0:
        potential = potential + rng.normal(loc=0.0, scale=float(noise_std), size=potential.shape)

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


def _validate_prediction_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Inversion predictions not found: {npz_path}")

    data = np.load(npz_path)
    if "points" not in data.files or "conductivity" not in data.files:
        raise KeyError(f"Expected points and conductivity in {npz_path}; got {data.files}")

    points = np.asarray(data["points"], dtype=np.float64)
    conductivity = _flatten(np.asarray(data["conductivity"], dtype=np.float64))
    potential = _flatten(np.asarray(data["potential"], dtype=np.float64)) if "potential" in data.files else None

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Inversion points must have shape (N, 3)")
    if conductivity.shape[0] != points.shape[0]:
        raise ValueError("Conductivity values must match the number of inversion points")
    if potential is not None and potential.shape[0] != points.shape[0]:
        raise ValueError("Potential values must match the number of inversion points")

    return points, conductivity, potential


def _plot_conductivity_dashboard(
    points: np.ndarray,
    conductivity: np.ndarray,
    output_path: Path,
    *,
    title: str,
    max_points: int,
) -> Path:
    plot_points, plot_sigma = _downsample(points, conductivity, max_points=max_points)
    vmin = float(np.min(conductivity))
    vmax = float(np.max(conductivity))

    fig = plt.figure(figsize=(12, 9), dpi=170, constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    scatter3d = ax3d.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        c=plot_sigma,
        cmap="viridis",
        s=9,
        alpha=0.85,
        linewidths=0.0,
        vmin=vmin,
        vmax=vmax,
    )
    ax3d.set_title("Volumen invertido")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    fig.colorbar(scatter3d, ax=ax3d, shrink=0.72, label="sigma")

    projections = [
        ("Proyeccion XY", 0, 1, "x", "y"),
        ("Proyeccion XZ", 0, 2, "x", "z"),
        ("Histograma sigma", None, None, "sigma", "frecuencia"),
    ]
    axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, (panel_title, x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        if x_idx is None or y_idx is None:
            ax.hist(conductivity, bins=35, color="#2a9d8f", edgecolor="#184e45", alpha=0.9)
            ax.set_title(panel_title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            continue

        scatter = ax.scatter(
            plot_points[:, x_idx],
            plot_points[:, y_idx],
            c=plot_sigma,
            cmap="viridis",
            s=12,
            alpha=0.9,
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(scatter, ax=ax, label="sigma")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _slice_indices(points: np.ndarray, level: float, half_width: float, min_points: int) -> np.ndarray:
    z_values = points[:, 2]
    mask = np.abs(z_values - level) <= half_width
    indices = np.flatnonzero(mask)
    if indices.shape[0] >= min_points or points.shape[0] <= min_points:
        return indices

    nearest_count = min(min_points, points.shape[0])
    return np.argsort(np.abs(z_values - level))[:nearest_count]


def _plot_depth_slices(
    points: np.ndarray,
    conductivity: np.ndarray,
    output_path: Path,
    *,
    levels: tuple[float, float, float] | None = None,
) -> Path:
    z_min = float(np.min(points[:, 2]))
    z_max = float(np.max(points[:, 2]))
    if levels is None:
        levels = tuple(float(v) for v in np.linspace(z_min, z_max, 5)[1:-1])

    z_span = max(z_max - z_min, 1e-9)
    half_width = 0.08 * z_span
    vmin = float(np.min(conductivity))
    vmax = float(np.max(conductivity))

    fig, axes = plt.subplots(1, len(levels), figsize=(12, 4), dpi=170, constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    last_scatter = None

    for ax, level in zip(axes_array, levels):
        idx = _slice_indices(points, float(level), half_width=half_width, min_points=80)
        slice_points = points[idx]
        slice_sigma = conductivity[idx]
        last_scatter = ax.scatter(
            slice_points[:, 0],
            slice_points[:, 1],
            c=slice_sigma,
            cmap="viridis",
            s=18,
            alpha=0.9,
            linewidths=0.0,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"z ~= {level:.2f}  n={idx.shape[0]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    if last_scatter is not None:
        fig.colorbar(last_scatter, ax=axes_array.tolist(), label="sigma")

    fig.suptitle("Cortes horizontales de conductividad")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_loss_history(output_root: Path, output_path: Path) -> Path | None:
    train_rows = _read_history(output_root / "train_loss_history.csv")
    invert_rows = _read_history(output_root / "invert_loss_history.csv")
    if not train_rows and not invert_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=170, constrained_layout=True)

    for rows, label, style in (
        (train_rows, "train total", "--"),
        (invert_rows, "invert total", "-"),
    ):
        if not rows:
            continue
        epochs = np.asarray([row["epoch"] for row in rows], dtype=np.float64)
        total = np.asarray([row["total"] for row in rows], dtype=np.float64)
        axes[0].semilogy(epochs, np.maximum(total, 1e-30), style, label=label)

    axes[0].set_title("Perdida total")
    axes[0].set_xlabel("epoca")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    if invert_rows:
        epochs = np.asarray([row["epoch"] for row in invert_rows], dtype=np.float64)
        for key in ("data", "pde", "bc", "reg", "flux"):
            if key not in invert_rows[0]:
                continue
            values = np.asarray([row[key] for row in invert_rows], dtype=np.float64)
            axes[1].semilogy(epochs, np.maximum(values, 1e-30), label=key)
        axes[1].legend()

    axes[1].set_title("Componentes de inversion")
    axes[1].set_xlabel("epoca")
    axes[1].set_ylabel("loss")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_observation_fit(output_root: Path, output_path: Path) -> tuple[Path | None, dict[str, float] | None]:
    fit_path = output_root / "invert_observation_fit.npz"
    if not fit_path.exists():
        return None, None

    data = np.load(fit_path)
    points = np.asarray(data["points"], dtype=np.float64)
    observed = _flatten(np.asarray(data["observed"], dtype=np.float64))
    predicted = _flatten(np.asarray(data["predicted"], dtype=np.float64))
    residual = _flatten(np.asarray(data["residual"], dtype=np.float64))
    metrics = _regression_metrics(observed, predicted)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=170, constrained_layout=True)

    low = float(min(np.min(observed), np.min(predicted)))
    high = float(max(np.max(observed), np.max(predicted)))
    axes[0].scatter(observed, predicted, s=16, alpha=0.82, linewidths=0.0, color="#264653")
    axes[0].plot([low, high], [low, high], color="#e76f51", linewidth=1.4)
    axes[0].set_title("Observado vs predicho")
    axes[0].set_xlabel("potencial observado")
    axes[0].set_ylabel("potencial predicho")

    scatter = axes[1].scatter(
        points[:, 0],
        points[:, 1],
        c=residual,
        cmap="coolwarm",
        s=18,
        alpha=0.9,
        linewidths=0.0,
    )
    axes[1].set_title("Residual espacial XY")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(scatter, ax=axes[1], label="predicho - observado")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path, metrics


def create_visualization_suite(
    output_root: str | Path,
    *,
    title: str = "Simulacion visual de inversion ERT",
    max_points: int = 5000,
) -> dict[str, Any]:
    """Create visual diagnostics for an existing inversion output folder."""
    root = Path(output_root)
    visualization_dir = root / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    points, conductivity, potential = _validate_prediction_arrays(root / "inversion_predictions.npz")

    figures: dict[str, str] = {}
    dashboard = _plot_conductivity_dashboard(
        points,
        conductivity,
        visualization_dir / "conductivity_dashboard.png",
        title=title,
        max_points=max_points,
    )
    figures["conductivity_dashboard"] = str(dashboard)

    slices = _plot_depth_slices(points, conductivity, visualization_dir / "conductivity_depth_slices.png")
    figures["conductivity_depth_slices"] = str(slices)

    loss_plot = _plot_loss_history(root, visualization_dir / "loss_history.png")
    if loss_plot is not None:
        figures["loss_history"] = str(loss_plot)

    observation_plot, observation_metrics = _plot_observation_fit(root, visualization_dir / "observation_fit.png")
    if observation_plot is not None:
        figures["observation_fit"] = str(observation_plot)

    metrics: dict[str, Any] = {
        "points": int(points.shape[0]),
        "conductivity": _array_stats(conductivity),
        "bounds": {
            "x": [float(np.min(points[:, 0])), float(np.max(points[:, 0]))],
            "y": [float(np.min(points[:, 1])), float(np.max(points[:, 1]))],
            "z": [float(np.min(points[:, 2])), float(np.max(points[:, 2]))],
        },
    }
    if potential is not None:
        metrics["potential"] = _array_stats(potential)
    if observation_metrics is not None:
        metrics["observation_fit"] = observation_metrics

    summary = {
        "output_root": str(root),
        "visualization_dir": str(visualization_dir),
        "figures": figures,
        "metrics": metrics,
    }
    save_json(summary, visualization_dir / "visualization_summary.json")
    return summary


def run_visual_inversion_simulation(
    *,
    config_dir: str | Path,
    experiment_name: str = "exp_visual_simulation",
    preset: str = "tiny",
    observation_count: int = 256,
    noise_std: float = 0.0,
    train_epochs: int | None = None,
    invert_epochs: int | None = None,
    measurement_points: int | None = None,
    title: str = "Simulacion visual de inversion ERT",
) -> dict[str, Any]:
    """Run a synthetic train->invert scenario and create visual diagnostics."""
    config = load_project_config(config_dir)
    apply_runtime_overrides(
        config,
        preset=preset,
        train_epochs=train_epochs,
        invert_epochs=invert_epochs,
        measurement_points=measurement_points,
    )

    project_root = Path(config["_meta"]["project_root"])
    output_root_rel = str(config["base"].get("paths", {}).get("output_root", "outputs"))
    output_root = project_root / output_root_rel / experiment_name

    training_summary = run_minimal_inverse(config=config, output_root=output_root, mode="train")

    training_predictions_path = output_root / "training_predictions.npz"
    if not training_predictions_path.exists():
        raise FileNotFoundError(f"Training predictions not found: {training_predictions_path}")

    project_seed = int(config["base"].get("project", {}).get("seed", 42))
    observations_path = output_root / "synthetic_visual_observations.csv"
    observations = build_synthetic_observations(
        training_predictions_path,
        observations_path,
        observation_count=int(observation_count),
        noise_std=float(noise_std),
        seed=project_seed,
    )

    inverse_cfg = _section(config["inverse"], "inversion")
    obs_cfg = inverse_cfg.setdefault("observations", {})
    obs_cfg["path"] = _safe_rel_path(observations_path, project_root)
    obs_cfg["delimiter"] = ","
    obs_cfg["skiprows"] = 1
    obs_cfg["point_columns"] = [0, 1, 2]
    obs_cfg["value_column"] = 3

    inversion_summary = run_minimal_inverse(config=config, output_root=output_root, mode="invert")
    visualization = create_visualization_suite(output_root, title=title)

    summary = {
        "mode": "visual_synthetic_train_then_invert",
        "scenario": {
            "description": "Forward PINN predictions are sampled as synthetic potential observations, then the inverse PINN reconstructs conductivity.",
            "preset": preset,
            "observation_count": int(observation_count),
            "noise_std": float(noise_std),
        },
        "output_root": str(output_root),
        "training": training_summary,
        "observations": observations,
        "inversion": inversion_summary,
        "visualization": visualization,
    }
    save_json(summary, output_root / "visual_inversion_simulation_summary.json")
    return summary
