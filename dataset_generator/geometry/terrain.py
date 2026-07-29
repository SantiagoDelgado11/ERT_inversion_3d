import numpy as np
from typing import Tuple, Dict

from dataset_generator.config.settings import DomainConfig, AnomalyConfig

class GeometryManager:
    def __init__(self, domain_cfg: DomainConfig):
        self.Lx = domain_cfg.x_length
        self.Ly = domain_cfg.y_length
        self.Lz = domain_cfg.z_length

    def generate_sphere(self, anomaly_cfg: AnomalyConfig) -> Dict[str, float]:
        """
        Generates the deterministic sphere anomaly based on config.
        Z axis points DOWN. Surface is z=0. Depth is z > 0.
        """
        R = anomaly_cfg.r
        
        # Ensure it fits in the domain
        if self.Lx <= 2 * R or self.Ly <= 2 * R or self.Lz <= 2 * R:
            raise ValueError("Domain is too small for the sphere radius.")
        
        cx, cy, cz = anomaly_cfg.x, anomaly_cfg.y, anomaly_cfg.z
        
        return {
            "x": cx,
            "y": cy,
            "z": cz,
            "r": R,
            "resistivity": anomaly_cfg.resistivity
        }
    
    def get_ground_truth_grid(self, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the 3D regular grid for ground truth export.
        Returns 1D arrays for x, y, z coordinates of the grid cell centers.
        """
        # Cell centers
        nx = int(np.round(self.Lx / dx))
        ny = int(np.round(self.Ly / dy))
        nz = int(np.round(self.Lz / dz))
        
        x_centers = np.linspace(dx/2, self.Lx - dx/2, nx)
        y_centers = np.linspace(dy/2, self.Ly - dy/2, ny)
        z_centers = np.linspace(dz/2, self.Lz - dz/2, nz)
        
        return x_centers, y_centers, z_centers
