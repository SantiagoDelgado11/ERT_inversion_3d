import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path

from dataset_generator.config.settings import SurveyConfig, DomainConfig

class SurveyGenerator:
    def __init__(self, survey_cfg: SurveyConfig, domain_cfg: DomainConfig):
        self.cfg = survey_cfg
        self.domain_cfg = domain_cfg
        self.electrodes = self._generate_electrodes()
        
    def _generate_electrodes(self) -> np.ndarray:
        """
        Generates 2D electrode grid on z=0 surface.
        Returns: (N_elec, 3) array of coordinates.
        """
        grid_width_x = (self.cfg.n_electrodes_x - 1) * self.cfg.spacing
        grid_width_y = (self.cfg.n_electrodes_y - 1) * self.cfg.spacing
        
        start_x = (self.domain_cfg.x_length - grid_width_x) / 2
        start_y = (self.domain_cfg.y_length - grid_width_y) / 2
        
        x = np.linspace(start_x, start_x + grid_width_x, self.cfg.n_electrodes_x)
        y = np.linspace(start_y, start_y + grid_width_y, self.cfg.n_electrodes_y)
        
        xv, yv = np.meshgrid(x, y, indexing='ij')
        zv = np.zeros_like(xv)
        
        electrodes = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
        return electrodes

    def get_geometric_factor(self, a_loc: np.ndarray, m_loc: np.ndarray, n_loc: np.ndarray) -> float:
        """
        Calculate analytic geometric factor for Pole-Dipole on a half-space.
        K = 2*pi / ( (1/AM) - (1/AN) )
        """
        r_am = np.linalg.norm(a_loc - m_loc)
        r_an = np.linalg.norm(a_loc - n_loc)
        
        if r_am < 1e-5 or r_an < 1e-5:
            return np.inf
            
        term = (1.0 / r_am) - (1.0 / r_an)
        if abs(term) < 1e-8:
            return np.inf
            
        return 2 * np.pi / term

    def _generate_combinations(self) -> List[Tuple[int, int, int]]:
        """
        Generates Pole-Dipole combinations (A, M, N indices) focusing on 3D coverage.
        """
        combinations = set()
        nx, ny = self.cfg.n_electrodes_x, self.cfg.n_electrodes_y
        
        def get_idx(ix, iy):
            if 0 <= ix < nx and 0 <= iy < ny:
                return ix * ny + iy
            return -1

        # 8 Azimuths for omnidirectional 3D coverage
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),   # Inlines & Crosslines
            (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonals
        ]

        # Use the configurable a_skip parameter to reduce current injection density systematically
        skip = self.cfg.a_skip
        
        for i_a in range(0, nx, skip):
            for j_a in range(0, ny, skip):
                a_idx = get_idx(i_a, j_a)
                
                for dx, dy in directions:
                    for a_sep in range(1, self.cfg.max_a_spacing + 1):
                        for n_spacing in range(1, self.cfg.max_n + 1):
                            
                            # Distance A -> M is n_spacing * a_sep
                            # Distance M -> N is a_sep
                            dist_am = n_spacing * a_sep
                            dist_mn = a_sep
                            
                            i_m = i_a + dx * dist_am
                            j_m = j_a + dy * dist_am
                            
                            i_n = i_m + dx * dist_mn
                            j_n = j_m + dy * dist_mn
                            
                            m_idx = get_idx(i_m, j_m)
                            n_idx = get_idx(i_n, j_n)
                            
                            if m_idx != -1 and n_idx != -1:
                                if a_idx != m_idx and m_idx != n_idx and a_idx != n_idx:
                                    combinations.add((a_idx, m_idx, n_idx))
                                    
        return list(combinations)

    def generate_survey(self) -> Dict[str, np.ndarray]:
        """
        Generates the full survey arrays.
        """
        combinations = self._generate_combinations()
        
        valid_a = []
        valid_m = []
        valid_n = []
        valid_k = []
        
        max_k = 50000.0  # Threshold to avoid numerical instability
        
        for a_idx, m_idx, n_idx in combinations:
            a_loc = self.electrodes[a_idx]
            m_loc = self.electrodes[m_idx]
            n_loc = self.electrodes[n_idx]
            
            k = self.get_geometric_factor(a_loc, m_loc, n_loc)
            if abs(k) < max_k:
                valid_a.append(a_idx)
                valid_m.append(m_idx)
                valid_n.append(n_idx)
                valid_k.append(k)
                
        return {
            "electrodes": self.electrodes,
            "a_idx": np.array(valid_a, dtype=int),
            "m_idx": np.array(valid_m, dtype=int),
            "n_idx": np.array(valid_n, dtype=int),
            "k_factor": np.array(valid_k, dtype=float)
        }

    def generate_report(self, survey_data: Dict[str, np.ndarray], output_dir: str):
        """
        Generates an objective evaluation report of the acquisition geometry.
        """
        a_idx = survey_data['a_idx']
        m_idx = survey_data['m_idx']
        n_idx = survey_data['n_idx']
        electrodes = survey_data['electrodes']
        
        total_elec = len(electrodes)
        total_meas = len(a_idx)
        unique_a = len(np.unique(a_idx))
        
        # Calculate pseudo-depths (L / 2 for Pole-Dipole roughly)
        a_loc = electrodes[a_idx]
        m_loc = electrodes[m_idx]
        n_loc = electrodes[n_idx]
        
        mn_midpoints = (m_loc + n_loc) / 2.0
        am_distances = np.linalg.norm(a_loc - m_loc, axis=1)
        pseudo_depths = am_distances / 2.0  # Approximation
        
        # Coverage metrics
        avg_depth = np.mean(pseudo_depths)
        max_depth = np.max(pseudo_depths)
        
        report_content = [
            "=== 3D ERT Acquisition Report ===",
            f"Total Electrodes: {total_elec}",
            f"Total Valid Measurements: {total_meas}",
            f"Unique Current Injectors (A): {unique_a} ({unique_a/total_elec*100:.1f}%)",
            "",
            "--- Spatial Coverage ---",
            f"Pseudo-depth Range: {np.min(pseudo_depths):.1f} m to {max_depth:.1f} m",
            f"Average Pseudo-depth: {avg_depth:.1f} m",
            f"Measurements targeting shallow (< {avg_depth/2:.1f}m): {np.sum(pseudo_depths < avg_depth/2)}",
            f"Measurements targeting deep (> {avg_depth*1.5:.1f}m): {np.sum(pseudo_depths > avg_depth*1.5)}",
            "",
            "--- Configuration Constraints Used ---",
            f"A Skip Factor: {self.cfg.a_skip}",
            f"Max N-factor: {self.cfg.max_n}",
            f"Max A-spacing: {self.cfg.max_a_spacing}",
        ]
        
        report_path = Path(output_dir) / "survey_report.txt"
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("\n".join(report_content))
        
        print(f"Acquisition report saved to {report_path}")
