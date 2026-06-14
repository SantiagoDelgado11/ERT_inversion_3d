"""Utilities for visualization and analysis of ERT PINN results."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_run_summary(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    inv_path = run_dir / "inversion_summary.json"
    if inv_path.exists():
        with open(inv_path, "r") as f:
            return json.load(f)
    train_path = run_dir / "training_summary.json"
    if train_path.exists():
        with open(train_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"No summary json found in {run_dir}")

def load_loss_history(run_dir: Path | str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    inv_path = run_dir / "invert_loss_history.csv"
    if inv_path.exists():
        return pd.read_csv(inv_path)
    train_path = run_dir / "train_loss_history.csv"
    if train_path.exists():
        return pd.read_csv(train_path)
    raise FileNotFoundError(f"No loss history csv found in {run_dir}")

def load_predictions(run_dir: Path | str) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    inv_path = run_dir / "inversion_predictions.npz"
    if inv_path.exists():
        return dict(np.load(inv_path))
    train_path = run_dir / "training_predictions.npz"
    if train_path.exists():
        return dict(np.load(train_path))
    raise FileNotFoundError(f"No predictions npz found in {run_dir}")

def plot_3d_scatter_slice(
    points: np.ndarray, 
    values: np.ndarray, 
    title: str, 
    output_path: Path, 
    slice_axis: str | None = None, 
    slice_value: float | None = None, 
    cmap: str = "viridis",
    val_name: str = "Value"
) -> None:
    """Plots a 2D slice of 3D point cloud data and saves it."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if slice_axis and slice_value is not None:
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(slice_axis.lower(), 2)
        # Find points close to the slice
        tolerance = 0.05
        mask = np.abs(points[:, axis_idx] - slice_value) < tolerance
        pts = points[mask]
        vals = values[mask]
        
        if len(pts) == 0:
            print(f"Warning: No points found near {slice_axis}={slice_value}")
            plt.close(fig)
            return
            
        # Determine the two axes to plot
        plot_axes = [i for i in range(3) if i != axis_idx]
        x_pts = pts[:, plot_axes[0]]
        y_pts = pts[:, plot_axes[1]]
        
        # Sort or just scatter
        sc = ax.scatter(x_pts, y_pts, c=vals, cmap=cmap, s=20, edgecolors='none')
        axes_labels = ["X", "Y", "Z"]
        ax.set_xlabel(axes_labels[plot_axes[0]])
        ax.set_ylabel(axes_labels[plot_axes[1]])
        ax.set_title(f"{title} (Slice {slice_axis.upper()} ~ {slice_value})")
        
        # Force equal aspect ratio to prevent stretching
        ax.set_aspect('equal', adjustable='box')
    else:
        # Default to plotting X-Z slice at y~0 if not specified
        mask = np.abs(points[:, 1]) < 0.1
        pts = points[mask]
        vals = values[mask]
        if len(pts) > 0:
            sc = ax.scatter(pts[:, 0], pts[:, 2], c=vals, cmap=cmap, s=20, edgecolors='none')
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_title(f"{title} (Central Y-Slice)")
            ax.set_aspect('equal', adjustable='box')
        else:
            # Fallback to pure 3D or all points 2D projection
            sc = ax.scatter(points[:, 0], points[:, 2], c=values, cmap=cmap, s=5, alpha=0.5, edgecolors='none')
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_title(f"{title} (All points projected to X-Z)")
            ax.set_aspect('equal', adjustable='box')
            
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(val_name)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
