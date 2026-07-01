"""
forward_solver.py

Provides an abstract generic interface for a discrete numerical forward solver 
(e.g., FVM, FEM using FEniCS, SimPEG, etc.).
This ensures the validation module is agnostic to the specific numerical backend,
using dependency injection to connect the PINN predictions with discrete physics.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any

class BaseForwardValidator(ABC):
    """
    Abstract Base Class for numerical forward solvers used to validate PINN ERT models.
    """
    
    def __init__(self, mesh_params: dict):
        """
        Initializes the solver with mesh and domain parameters.
        
        Args:
            mesh_params (dict): Dictionary defining mesh limits, resolutions, and properties.
        """
        self.mesh_params = mesh_params
        self.mesh = self._build_mesh()
        
    @abstractmethod
    def _build_mesh(self) -> Any:
        """
        Constructs the discrete mesh.
        
        Returns:
            Any: The specific mesh object for the chosen backend.
        """
        pass
        
    @abstractmethod
    def get_evaluation_points(self) -> np.ndarray:
        """
        Extracts the coordinates of nodes or cell centers from the discrete mesh.
        These points are where the continuous PINN `conductivity_net` must be evaluated.
        
        Returns:
            np.ndarray: Array of shape (N_points, 3) representing 3D spatial coordinates.
        """
        pass

    @abstractmethod
    def assemble_gaussian_source(self, current_electrodes: Tuple[np.ndarray, np.ndarray], width: float) -> np.ndarray:
        """
        Constructs the source term for the PDE using a volumetric Gaussian distribution 
        to approximate the Dirac delta functions at the injection and extraction electrodes.
        
        Physical Purpose: 
        In ERT, current is injected at a point A and extracted at point B (a dipole). 
        True Dirac deltas cause singularities in numerical solvers and PINNs. By using a 
        Gaussian approximation, we smear the point source over a small volume defined by `width`.
        This matches the stabilized formulation of the PDE optimized by the PINN.
        
        Args:
            current_electrodes: Tuple of (pos_A, pos_B) where each is a 3D coordinate array.
            width: Standard deviation (spread) of the Gaussian source.
            
        Returns:
            np.ndarray: The assembled source vector/field on the discrete mesh.
        """
        pass

    @abstractmethod
    def apply_boundary_conditions(self) -> None:
        """
        Imposes boundary conditions on the discrete system.
        
        Physical Purpose:
        - Neumann BCs (zero flux) at the surface/ground boundary to represent the air-earth interface
          where no current escapes.
        - Dirichlet or asymptotic boundary conditions at infinite/far-field boundaries where 
          the electric potential decays to zero.
          
        This must perfectly match the PDEs boundary formulation within the PINN loss function.
        """
        pass

    @abstractmethod
    def solve_pde(self, conductivity_field: np.ndarray, source_term: np.ndarray) -> np.ndarray:
        """
        Solves the ERT forward problem: -∇ · (σ * ∇φ) = I * (δ_A - δ_B)
        
        Args:
            conductivity_field (np.ndarray): The conductivity evaluated at mesh points 
                                             (often provided by the PINN `conductivity_net`).
            source_term (np.ndarray): The stabilized Gaussian source term.
            
        Returns:
            np.ndarray: The discrete electric potential field (φ) solved over the mesh.
        """
        pass
