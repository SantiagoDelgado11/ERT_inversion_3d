"""
metrics.py

Provides mathematical error metrics for evaluating continuous PINN predictions
against true models or discrete numerical solutions. Includes standard regression 
metrics and image quality assessment metrics for 2D/3D fields.
"""

import numpy as np
from typing import Union, Dict, Any, Optional

def rmse(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((true_values - pred_values) ** 2)))

def relative_l2_error(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """Calculates the relative L2 error norm."""
    num = np.linalg.norm(true_values - pred_values)
    den = np.linalg.norm(true_values)
    return float(num / den) if den != 0 else float('inf')

def mae(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(true_values - pred_values)))

def max_error(true_values: np.ndarray, pred_values: np.ndarray) -> float:
    """Calculates the Maximum Absolute Error."""
    return float(np.max(np.abs(true_values - pred_values)))

def calculate_psnr(true_values: np.ndarray, pred_values: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR).
    """
    if data_range is None:
        data_range = float(true_values.max() - true_values.min())
    
    mse = np.mean((true_values - pred_values) ** 2)
    if mse == 0:
        return float('inf')
        
    return float(10 * np.log10((data_range ** 2) / mse))

def calculate_ssim(true_values: np.ndarray, pred_values: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Calculates the Structural Similarity Index (SSIM).
    Assumes inputs are reshaped to the proper 2D or 3D grid.
    Requires scikit-image to be installed.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        raise ImportError("scikit-image is required for SSIM calculation.")
        
    if data_range is None:
        data_range = float(true_values.max() - true_values.min())
        
    # structural_similarity supports N-D arrays
    return float(structural_similarity(true_values, pred_values, data_range=data_range))

def compute_all_metrics(true_values: np.ndarray, pred_values: np.ndarray, grid_shape: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Computes a consolidated dictionary of all metrics.
    If grid_shape is provided, reshapes arrays for spatial metrics (PSNR, SSIM).
    """
    metrics: Dict[str, Any] = {
        "rmse": rmse(true_values, pred_values),
        "rel_l2": relative_l2_error(true_values, pred_values),
        "mae": mae(true_values, pred_values),
        "max_error": max_error(true_values, pred_values)
    }
    
    if grid_shape is not None:
        try:
            t_grid = true_values.reshape(grid_shape)
            p_grid = pred_values.reshape(grid_shape)
            data_range = float(t_grid.max() - t_grid.min())
            metrics["psnr"] = calculate_psnr(t_grid, p_grid, data_range)
            metrics["ssim"] = calculate_ssim(t_grid, p_grid, data_range)
        except Exception as e:
            metrics["psnr"] = None
            metrics["ssim"] = None
            metrics["spatial_metric_error"] = str(e)
            
    return metrics
