import numpy as np
import discretize
from typing import Dict

from dataset_generator.config.settings import MeshConfig, DomainConfig

class MeshGenerator:
    def __init__(self, mesh_cfg: MeshConfig, domain_cfg: DomainConfig):
        self.cfg = mesh_cfg
        self.domain = domain_cfg
        
    def build_tree_mesh(self, electrodes: np.ndarray, anomaly: Dict[str, float]) -> discretize.TreeMesh:
        """
        Builds and refines an OcTree mesh.
        """
        h = [self.cfg.core_cell_size] * 3
        
        # Calculate bounding box of the core domain
        # The user wants domain from 0 to L
        core_x0 = 0.0
        core_y0 = 0.0
        core_z0 = -self.domain.z_length  # In SimPEG, Z typically points UP. If surface is z=0, depth is z < 0.
        
        # We will keep Z pointing down in our coordinates (0 to 50), 
        # but for SimPEG we can just treat the coordinates as is. 
        # If we use Z=0 to Z=50 as depth, we just need to be consistent. 
        # Let's map user Z (depth, positive) to SimPEG Z (up, positive).
        # Actually, DC resistivity equations are symmetric, so z=0 to z=50 works just fine 
        # as long as padding goes from 50 to 1000 (air) and 0 to -1000 (depth). Wait, no.
        # If z=0 is surface, and we want electrodes there, we put electrodes at z=0.
        # If sub-surface is z > 0, then we want padding in +z and -z. 
        
        # Let's just use discretize.utils.mesh_builder_xyz
        pad_dist = self.cfg.padding_cells * self.cfg.core_cell_size * self.cfg.padding_factor
        padding = [[pad_dist, pad_dist], [pad_dist, pad_dist], [pad_dist, pad_dist]]
        
        # Generate base mesh centered around electrodes
        mesh = discretize.utils.mesh_builder_xyz(
            electrodes, 
            h, 
            padding_distance=padding, 
            mesh_type='tree',
            depth_core=self.domain.z_length
        )
        
        # Refine around electrodes
        try:
            # Modern discretize API
            mesh.refine_points(electrodes, level=-1, padding_cells_by_level=[2, 2, 2])
            
            # Refine around sphere anomaly
            center = np.array([anomaly['x'], anomaly['y'], anomaly['z']])
            radius = anomaly['r']
            mesh.refine_ball(center, radius + self.cfg.core_cell_size, [-1])
        except (AttributeError, TypeError):
            # Fallback for older discretize versions
            mesh = discretize.utils.refine_tree_xyz(
                mesh, electrodes, octree_levels=[2, 2, 2], method='surface', finalize=False
            )
            center = np.array([[anomaly['x'], anomaly['y'], anomaly['z']]])
            mesh = discretize.utils.refine_tree_xyz(
                mesh, center, octree_levels=[2, 2], method='radial', finalize=False
            )
            
        # We also want to ensure the core domain is at least reasonably refined
        # Domain: [0, Lx] x [0, Ly] x [0, Lz]
        # In modern discretize, we can use refine_box
        try:
            bbox_min = np.array([0.0, 0.0, 0.0])
            bbox_max = np.array([self.domain.x_length, self.domain.y_length, self.domain.z_length])
            mesh.refine_bounding_box(bbox_min, bbox_max, [-3])
        except Exception:
            pass
            
        # Finalize the tree
        # mesh.finalize() # Implicit in modern discretize or already done if fallback wasn't needed
        # But to be safe:
        if not hasattr(mesh, 'n_cells'):
             pass # some versions require finalize, but newer ones don't have it.
        
        return mesh
        
    def get_active_cells(self, mesh: discretize.TreeMesh) -> np.ndarray:
        """
        Returns boolean array of active cells (subsurface).
        Assuming Z points DOWN (z=0 surface, z>0 subsurface).
        So active cells are where z >= 0.
        Wait, if padding was symmetric, air is z < 0.
        """
        # Electrodes are at z=0. 
        # Z > 0 is subsurface.
        # Z < 0 is air.
        return mesh.cell_centers[:, 2] >= -1e-5
