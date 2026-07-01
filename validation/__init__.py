"""
Validation module for 3D ERT PINN evaluation.
Provides tools to mathematically and visually evaluate PINN performance
against both synthetic ground-truth targets and discrete numerical solvers.
"""

from .metrics import compute_all_metrics, rmse, mae
from .plots import plot_conductivity_comparison, plot_error_map, plot_error_histogram, plot_1d_profile
from .forward_solver import BaseForwardValidator
from .evaluator import ValidationPipeline

__all__ = [
    "ValidationPipeline",
    "BaseForwardValidator",
    "compute_all_metrics",
    "rmse",
    "mae",
    "plot_conductivity_comparison",
    "plot_error_map",
    "plot_error_histogram",
    "plot_1d_profile"
]
