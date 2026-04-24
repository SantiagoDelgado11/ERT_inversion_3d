"""Create a quick PNG visualization from inversion prediction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    if "points" in data.files and "conductivity" in data.files:
        points = np.asarray(data["points"], dtype=np.float64)
        sigma = np.asarray(data["conductivity"], dtype=np.float64)
        return points, sigma

    if "train_points" in data.files and "train_conductivity_pred" in data.files:
        points = np.asarray(data["train_points"], dtype=np.float64)
        sigma = np.asarray(data["train_conductivity_pred"], dtype=np.float64)
        return points, sigma

    expected = "{points, conductivity} or {train_points, train_conductivity_pred}"
    raise KeyError(f"Unsupported NPZ keys in {npz_path}. Expected {expected}, got {data.files}")


def _downsample(points: np.ndarray, sigma: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    n_points = points.shape[0]
    if n_points <= max_points:
        return points, sigma

    idx = np.linspace(0, n_points - 1, max_points, dtype=np.int64)
    return points[idx], sigma[idx]


def create_figure(npz_path: Path, output_path: Path, title: str) -> Path:
    points, sigma = _resolve_arrays(npz_path)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3)")

    sigma = np.asarray(sigma).reshape(-1)
    if sigma.shape[0] != points.shape[0]:
        raise ValueError("Conductivity array length must match number of points")

    points_plot, sigma_plot = _downsample(points, sigma)

    fig = plt.figure(figsize=(12, 5), dpi=160)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    scatter = ax3d.scatter(
        points_plot[:, 0],
        points_plot[:, 1],
        points_plot[:, 2],
        c=sigma_plot,
        cmap="viridis",
        s=10,
        alpha=0.9,
        linewidths=0.0,
    )
    ax3d.set_title("Conductividad reconstruida")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    colorbar = fig.colorbar(scatter, ax=ax3d, shrink=0.72, pad=0.08)
    colorbar.set_label("sigma")

    ax_hist = fig.add_subplot(1, 2, 2)
    ax_hist.hist(sigma, bins=30, color="#2a9d8f", edgecolor="#1b4332", alpha=0.9)
    ax_hist.set_title("Distribucion de conductividad")
    ax_hist.set_xlabel("sigma")
    ax_hist.set_ylabel("frecuencia")

    fig.suptitle(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize inversion predictions as a PNG file")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/exp_default/inversion_predictions.npz",
        help="Path to inversion_predictions.npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/exp_default/inversion_result.png",
        help="Path for output PNG image",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Resultado de inversion ERT",
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    saved = create_figure(input_path, output_path, title=str(args.title))
    print(f"Saved inversion figure to: {saved}")


if __name__ == "__main__":
    main()
