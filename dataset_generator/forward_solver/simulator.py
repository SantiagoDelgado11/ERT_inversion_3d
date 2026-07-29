import numpy as np
import discretize
import simpeg.electromagnetics.static.resistivity as dc
from simpeg import maps
from typing import Dict

from dataset_generator.config.settings import AppConfig

class ForwardSolver:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        
    def build_physical_model(self, mesh: discretize.TreeMesh, anomaly: Dict[str, float], active_cells: np.ndarray) -> np.ndarray:
        """
        Creates the conductivity model on active cells.
        Returns:
            active_model: Conductivity values for active cells.
        """
        bg_sigma = 1.0 / self.cfg.domain.bg_resistivity
        sphere_sigma = 1.0 / anomaly['resistivity']
        
        # Initialize full mesh with background
        sigma = np.ones(mesh.n_cells) * bg_sigma
        
        # Identify sphere cells
        cc = mesh.cell_centers
        dist = np.sqrt((cc[:, 0] - anomaly['x'])**2 + 
                       (cc[:, 1] - anomaly['y'])**2 + 
                       (cc[:, 2] - anomaly['z'])**2)
        is_sphere = dist <= anomaly['r']
        sigma[is_sphere] = sphere_sigma
        
        # We only pass active cells to the solver
        return sigma[active_cells]

    def run_simulation(self, mesh: discretize.TreeMesh, active_cells: np.ndarray, 
                       model_active: np.ndarray, survey_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Groups sources, runs SimPEG simulation, and returns updated survey data with voltages.
        """
        electrodes = survey_data['electrodes']
        
        unique_a = np.unique(survey_data['a_idx'])
        src_list = []
        
        # We must reorder our arrays to match SimPEG's internal ordering (by source, then receiver)
        ordered_a = []
        ordered_m = []
        ordered_n = []
        ordered_k = []
        
        for a_id in unique_a:
            mask = survey_data['a_idx'] == a_id
            m_indices = survey_data['m_idx'][mask]
            n_indices = survey_data['n_idx'][mask]
            k_factors = survey_data['k_factor'][mask]
            
            ordered_a.extend(np.full(len(m_indices), a_id))
            ordered_m.extend(m_indices)
            ordered_n.extend(n_indices)
            ordered_k.extend(k_factors)
            
            m_locs = electrodes[m_indices]
            n_locs = electrodes[n_indices]
            a_loc = electrodes[a_id]
            
            rx = dc.receivers.Dipole(m_locs, n_locs)
            src = dc.sources.Pole([rx], location=a_loc)
            src_list.append(src)
            
        # Update survey data with ordered arrays
        survey_data['a_idx'] = np.array(ordered_a, dtype=int)
        survey_data['m_idx'] = np.array(ordered_m, dtype=int)
        survey_data['n_idx'] = np.array(ordered_n, dtype=int)
        survey_data['k_factor'] = np.array(ordered_k, dtype=float)
        
        simpeg_survey = dc.Survey(src_list)
        
        # Mapping: active cells -> full mesh. Inactive cells get air conductivity.
        try:
            actmap = maps.InjectActiveCells(mesh, active_cells=active_cells, value_inactive=1e-8)
        except TypeError:
            # Fallback for older SimPEG versions
            actmap = maps.InjectActiveCells(mesh, indActive=active_cells, valInactive=1e-8)
        
        # Setup Simulation
        # We use a good solver if available
        try:
            from pymatsolver import Pardiso as Solver
        except ImportError:
            try:
                import pydiso.mkl_solver
                from simpeg.utils import SolverLU as Solver
            except ImportError:
                from simpeg.utils import SolverLU as Solver # Fallback to scipy Sparse LU (slow)
                
        simulation = dc.Simulation3DCellCentered(
            mesh, 
            survey=simpeg_survey, 
            sigmaMap=actmap, 
            solver=Solver
        )
        
        # Predict data (Potentials normalized by injected current: V / I = R)
        # SimPEG static resistivity sources assume I = 1 A by default.
        # So dpred is actually Resistance (R).
        dpred = simulation.dpred(model_active)
        
        # Scale to true voltage
        I = self.cfg.survey.injected_current
        V = dpred * I
        R = dpred
        rho_a = R * survey_data['k_factor']
        
        survey_data['voltage'] = V
        survey_data['current'] = np.full_like(V, I)
        survey_data['resistance'] = R
        survey_data['apparent_resistivity'] = rho_a
        
        return survey_data
