"""
evaluator.py

Central evaluation pipeline for the ERT PINN model.
Orchestrates the evaluation without training, calling metrics, plots, 
and optionally the forward solver.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .metrics import compute_all_metrics
from .plots import (
    plot_conductivity_comparison, 
    plot_error_map, 
    plot_error_histogram, 
    plot_1d_profile
)
from .forward_solver import BaseForwardValidator

class ValidationPipeline:
    """
    Pipeline for orchestrating the validation of an ERT PINN model.
    Evaluates continuous neural networks against observed data and/or 
    synthetic true models over a discrete mesh.
    """
    
    def __init__(
        self, 
        conductivity_net: nn.Module, 
        potential_net: nn.Module,
        forward_solver: Optional[BaseForwardValidator] = None,
        device: str = "cpu"
    ):
        """
        Initializes the pipeline with trained networks and an optional discrete solver.
        
        Args:
            conductivity_net (nn.Module): Trained PINN representing σ(x,y,z).
            potential_net (nn.Module): Trained PINN representing φ(x,y,z).
            forward_solver (BaseForwardValidator, optional): Injected discrete solver.
            device (str): Device to evaluate tensors on (e.g., 'cpu', 'cuda').
        """
        self.conductivity_net = conductivity_net.to(device).eval()
        self.potential_net = potential_net.to(device).eval()
        self.forward_solver = forward_solver
        self.device = device
        
    @torch.no_grad()
    def evaluate_conductivity(
        self, 
        points: np.ndarray, 
        true_conductivity: Optional[np.ndarray] = None,
        grid_shape: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates the predicted conductivity at given points and compares with true conductivity.
        
        Args:
            points (np.ndarray): Coordinates (N, 3) to evaluate the network.
            true_conductivity (np.ndarray, optional): True conductivity array for comparison.
            grid_shape (tuple, optional): Spatial dimensions for image metrics/plots.
            
        Returns:
            Dict[str, Any]: Consolidated dictionary of metrics and figures.
        """
        pts_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
        pred_cond = self.conductivity_net(pts_tensor).cpu().numpy().flatten()
        
        results: Dict[str, Any] = {
            "predictions": pred_cond,
            "metrics": {},
            "figures": {}
        }
        
        if true_conductivity is not None:
            true_cond = true_conductivity.flatten()
            results["metrics"] = compute_all_metrics(true_cond, pred_cond, grid_shape)
            
            # Generate plots if grid shape is provided for a 3D field
            if grid_shape is not None and len(grid_shape) == 3:
                mid_slice = grid_shape[2] // 2
                results["figures"]["cond_comparison"] = plot_conductivity_comparison(
                    true_cond, pred_cond, grid_shape, slice_idx=mid_slice
                )
                results["figures"]["error_map_abs"] = plot_error_map(
                    true_cond, pred_cond, grid_shape, slice_idx=mid_slice, relative=False
                )
                results["figures"]["error_map_rel"] = plot_error_map(
                    true_cond, pred_cond, grid_shape, slice_idx=mid_slice, relative=True
                )
                mid_cut = (grid_shape[0] // 2, grid_shape[1] // 2)
                results["figures"]["profile_1d"] = plot_1d_profile(
                    true_cond, pred_cond, grid_shape, line_indices=mid_cut
                )
            
            results["figures"]["error_hist"] = plot_error_histogram(true_cond, pred_cond)
            
        return results

    @torch.no_grad()
    def evaluate_forward_physics(
        self, 
        current_electrodes: Tuple[np.ndarray, np.ndarray], 
        source_width: float = 0.05
    ) -> Dict[str, Any]:
        """
        Evaluates the physics by mapping PINN conductivity to the discrete solver, 
        solving the PDE numerically, and comparing with the PINN's potential_net.
        
        Args:
            current_electrodes (tuple): (pos_A, pos_B) injection/extraction positions.
            source_width (float): Gaussian spread for the source term approximation.
            
        Returns:
            Dict[str, Any]: Metrics and potential comparisons between PINN and discrete solver.
        """
        if self.forward_solver is None:
            raise ValueError("A forward_solver instance must be injected to evaluate physics.")
            
        # 1. Extract nodes/cell centers from discrete solver
        mesh_points = self.forward_solver.get_evaluation_points()
        
        # 2. Evaluate PINN conductivity and potential at those mesh points
        pts_tensor = torch.tensor(mesh_points, dtype=torch.float32, device=self.device)
        pinn_cond = self.conductivity_net(pts_tensor).cpu().numpy().flatten()
        pinn_pot = self.potential_net(pts_tensor).cpu().numpy().flatten()
        
        # 3. Setup solver physics (Source and BCs)
        source_term = self.forward_solver.assemble_gaussian_source(current_electrodes, source_width)
        self.forward_solver.apply_boundary_conditions()
        
        # 4. Solve the discrete PDE using PINN's conductivity
        discrete_pot = self.forward_solver.solve_pde(pinn_cond, source_term)
        
        # 5. Compare discrete potential vs PINN potential
        metrics = compute_all_metrics(discrete_pot, pinn_pot)
        
        fig_hist = plot_error_histogram(discrete_pot, pinn_pot)
        
        return {
            "pinn_potential": pinn_pot,
            "discrete_potential": discrete_pot,
            "physics_metrics": metrics,
            "figures": {
                "potential_error_hist": fig_hist
            }
        }
