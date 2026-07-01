"""
plots.py

Handles generation of Matplotlib figures for visual validation.
Functions return Figure objects for export without calling plt.show().
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def plot_conductivity_comparison(
    true_cond: np.ndarray, 
    pred_cond: np.ndarray, 
    grid_shape: Tuple[int, ...], 
    slice_idx: int, 
    axis: int = 2
) -> plt.Figure:
    """
    Plots true vs predicted conductivity map for a specific 2D slice.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    t_grid = true_cond.reshape(grid_shape)
    p_grid = pred_cond.reshape(grid_shape)
    
    if axis == 0:
        t_slice = t_grid[slice_idx, :, :]
        p_slice = p_grid[slice_idx, :, :]
    elif axis == 1:
        t_slice = t_grid[:, slice_idx, :]
        p_slice = p_grid[:, slice_idx, :]
    else:
        t_slice = t_grid[:, :, slice_idx]
        p_slice = p_grid[:, :, slice_idx]
        
    vmin = min(t_slice.min(), p_slice.min())
    vmax = max(t_slice.max(), p_slice.max())
    
    im1 = axes[0].imshow(t_slice, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title('True Conductivity')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(p_slice, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title('Predicted (PINN) Conductivity')
    fig.colorbar(im2, ax=axes[1])
    
    fig.tight_layout()
    plt.close(fig) # Prevent display if in interactive environment
    return fig

def plot_error_map(
    true_vals: np.ndarray, 
    pred_vals: np.ndarray, 
    grid_shape: Tuple[int, ...], 
    slice_idx: int, 
    axis: int = 2,
    relative: bool = False
) -> plt.Figure:
    """
    Plots the absolute or relative error map.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    t_grid = true_vals.reshape(grid_shape)
    p_grid = pred_vals.reshape(grid_shape)
    
    if axis == 0:
        t_slice = t_grid[slice_idx, :, :]
        p_slice = p_grid[slice_idx, :, :]
    elif axis == 1:
        t_slice = t_grid[:, slice_idx, :]
        p_slice = p_grid[:, slice_idx, :]
    else:
        t_slice = t_grid[:, :, slice_idx]
        p_slice = p_grid[:, :, slice_idx]
        
    if relative:
        error_slice = np.abs(t_slice - p_slice) / (np.abs(t_slice) + 1e-10)
        title = "Relative Error Map"
    else:
        error_slice = np.abs(t_slice - p_slice)
        title = "Absolute Error Map"
        
    im = ax.imshow(error_slice, cmap='inferno')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_error_histogram(true_vals: np.ndarray, pred_vals: np.ndarray) -> plt.Figure:
    """
    Plots a histogram of the point-wise absolute errors.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    errors = np.abs(true_vals - pred_vals).flatten()
    
    ax.hist(errors, bins=50, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of Absolute Errors')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_1d_profile(
    true_vals: np.ndarray, 
    pred_vals: np.ndarray, 
    grid_shape: Tuple[int, ...], 
    line_indices: Tuple[int, int], 
    axis: int = 2
) -> plt.Figure:
    """
    Plots a 1D comparative profile across a specific line cut in the domain.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    t_grid = true_vals.reshape(grid_shape)
    p_grid = pred_vals.reshape(grid_shape)
    
    idx1, idx2 = line_indices
    
    if axis == 0:
        t_line = t_grid[:, idx1, idx2]
        p_line = p_grid[:, idx1, idx2]
    elif axis == 1:
        t_line = t_grid[idx1, :, idx2]
        p_line = p_grid[idx1, :, idx2]
    else:
        t_line = t_grid[idx1, idx2, :]
        p_line = p_grid[idx1, idx2, :]
        
    ax.plot(t_line, label='True', linestyle='--', marker='o')
    ax.plot(p_line, label='Predicted (PINN)', alpha=0.7)
    
    ax.set_title(f'1D Profile (Cut at indices {idx1}, {idx2})')
    ax.set_xlabel('Grid Index along cut axis')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.close(fig)
    return fig
