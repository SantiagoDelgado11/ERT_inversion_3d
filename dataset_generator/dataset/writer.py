import numpy as np
import os
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict

from dataset_generator.config.settings import AppConfig

class HDF5Writer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
    def _interpolate_to_grid(self, mesh, active_model, gt_grid_x, gt_grid_y, gt_grid_z):
        """
        Interpolates the TreeMesh conductivity model onto a regular 3D grid.
        Returns a 3D tensor of conductivities corresponding to the grid points.
        """
        full_model = np.ones(mesh.n_cells) * (1.0 / self.cfg.domain.bg_resistivity)
        active_indices = mesh.cell_centers[:, 2] >= -1e-5
        full_model[active_indices] = active_model
        
        xv, yv, zv = np.meshgrid(gt_grid_x, gt_grid_y, gt_grid_z, indexing='ij')
        pts = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
        
        try:
            P = mesh.get_interpolation_matrix(pts, locType='CC')
            grid_vals = P * full_model
        except Exception:
            # Fallback if get_interpolation_matrix fails
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(mesh.cell_centers, full_model)
            grid_vals = interp(pts)
            
        # Reshape to 3D tensor: (nx, ny, nz)
        tensor_3d = grid_vals.reshape((len(gt_grid_x), len(gt_grid_y), len(gt_grid_z)))
        return tensor_3d

    def save_campaign(self, survey_data: Dict[str, np.ndarray], 
                      anomaly: Dict[str, float], active_model: np.ndarray,
                      mesh, gt_grid: tuple):
        
        filename = Path(self.cfg.output_dir) / "campaign.h5"
        
        with h5py.File(filename, 'w') as h5f:
            # --- 1. METADATA ---
            meta_grp = h5f.create_group("metadata")
            meta_grp.attrs['bg_resistivity'] = self.cfg.domain.bg_resistivity
            meta_grp.attrs['sphere_x'] = anomaly['x']
            meta_grp.attrs['sphere_y'] = anomaly['y']
            meta_grp.attrs['sphere_z'] = anomaly['z']
            meta_grp.attrs['sphere_r'] = anomaly['r']
            meta_grp.attrs['sphere_rho'] = anomaly['resistivity']
            
            # --- 2. ELECTRODES ---
            electrodes = survey_data['electrodes']
            h5f.create_dataset("electrode_positions", data=electrodes)
            
            # --- 3. SURVEY CONFIG & MEASUREMENTS ---
            meas_grp = h5f.create_group("measurements")
            
            a_idx = survey_data['a_idx']
            m_idx = survey_data['m_idx']
            n_idx = survey_data['n_idx']
            
            # Constraints: B_idx forced to -1
            b_idx = np.full_like(a_idx, -1, dtype=np.int32)
            
            meas_grp.create_dataset("A_idx", data=a_idx)
            meas_grp.create_dataset("B_idx", data=b_idx)
            meas_grp.create_dataset("M_idx", data=m_idx)
            meas_grp.create_dataset("N_idx", data=n_idx)
            
            # Coordinates
            a_loc = electrodes[a_idx]
            m_loc = electrodes[m_idx]
            n_loc = electrodes[n_idx]
            # Constraints: b_loc saved as NaN
            b_loc = np.full_like(a_loc, np.nan)
            
            meas_grp.create_dataset("A_loc", data=a_loc)
            meas_grp.create_dataset("B_loc", data=b_loc)
            meas_grp.create_dataset("M_loc", data=m_loc)
            meas_grp.create_dataset("N_loc", data=n_loc)
            
            # Data
            meas_grp.create_dataset("V", data=survey_data['voltage'])
            meas_grp.create_dataset("I", data=survey_data['current'])
            meas_grp.create_dataset("R", data=survey_data['resistance'])
            meas_grp.create_dataset("K", data=survey_data['k_factor'])
            meas_grp.create_dataset("Rho_a", data=survey_data['apparent_resistivity'])
            
            # --- 4. GROUND TRUTH CONDUCTIVITY ---
            gt_x, gt_y, gt_z = gt_grid
            gt_cond = self._interpolate_to_grid(mesh, active_model, gt_x, gt_y, gt_z)
            
            gt_grp = h5f.create_group("ground_truth_conductivity")
            gt_grp.create_dataset("conductivity_tensor", data=gt_cond)
            gt_grp.create_dataset("grid_x", data=gt_x)
            gt_grp.create_dataset("grid_y", data=gt_y)
            gt_grp.create_dataset("grid_z", data=gt_z)
            
            # --- 5. MESH (Optional Raw Data) ---
            mesh_grp = h5f.create_group("mesh")
            mesh_grp.create_dataset("cell_centers", data=mesh.cell_centers)
            # Store active model array for completeness
            mesh_grp.create_dataset("active_cells", data=(mesh.cell_centers[:, 2] >= -1e-5))
            mesh_grp.create_dataset("active_model_conductivity", data=active_model)
            
        print(f"Campaign saved successfully to {filename}")
        
        # --- 6. EXPORT MEASUREMENTS TO CSV ---
        csv_filename = Path(self.cfg.output_dir) / "measurements.csv"
        df_meas = pd.DataFrame({
            'A_idx': a_idx, 'B_idx': b_idx, 'M_idx': m_idx, 'N_idx': n_idx,
            'A_x': a_loc[:,0], 'A_y': a_loc[:,1], 'A_z': a_loc[:,2],
            'B_x': b_loc[:,0], 'B_y': b_loc[:,1], 'B_z': b_loc[:,2],
            'M_x': m_loc[:,0], 'M_y': m_loc[:,1], 'M_z': m_loc[:,2],
            'N_x': n_loc[:,0], 'N_y': n_loc[:,1], 'N_z': n_loc[:,2],
            'V': survey_data['voltage'],
            'I': survey_data['current'],
            'R': survey_data['resistance'],
            'K': survey_data['k_factor'],
            'Rho_a': survey_data['apparent_resistivity']
        })
        df_meas.to_csv(csv_filename, index=False)
        print(f"Measurements CSV saved successfully to {csv_filename}")
