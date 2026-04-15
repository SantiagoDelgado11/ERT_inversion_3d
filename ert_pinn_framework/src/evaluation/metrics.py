"""Standard metrics used to evaluate PINN and inversion outputs."""

from __future__ import annotations

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def relative_l2_error(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-12) -> float:
    """Relative L2 error: ||y_pred - y_true||_2 / ||y_true||_2."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    num = np.linalg.norm(yp - yt)
    den = np.linalg.norm(yt) + eps
    return float(num / den)


def summary_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Bundle common regression metrics in one dictionary."""
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "relative_l2": relative_l2_error(y_true, y_pred),
    }
