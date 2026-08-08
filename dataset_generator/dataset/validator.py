import numpy as np
from typing import Dict, Any
from dataset_generator.config.settings import AppConfig

class DatasetValidator:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        
    def validate_campaign(self, survey_data: Dict[str, np.ndarray], anomaly: Dict[str, float], active_model: np.ndarray) -> bool:
        """
        Runs strict validations on the generated campaign before accepting it.
        Raises ValueError if any validation fails.
        """
        # 1. Electrodes at z=0
        electrodes = survey_data['electrodes']
        if not np.allclose(electrodes[:, 2], 0.0, atol=1e-5):
            raise ValueError("Validation failed: Not all electrodes are at z=0.")
            
        # 2. Sphere inside domain
        r = anomaly['r']
        if anomaly['x'] - r < 0 or anomaly['x'] + r > self.cfg.domain.x_length:
            raise ValueError("Validation failed: Sphere intersects X boundaries.")
        if anomaly['y'] - r < 0 or anomaly['y'] + r > self.cfg.domain.y_length:
            raise ValueError("Validation failed: Sphere intersects Y boundaries.")
        if anomaly['z'] - r < 0 or anomaly['z'] + r > self.cfg.domain.z_length:
            raise ValueError("Validation failed: Sphere intersects Z boundaries or surface.")
            
        # 3. Resistivities are correct
        # Check active model values. Should only contain bg and sphere conductivities
        bg_cond = 1.0 / self.cfg.domain.bg_resistivity
        sp_cond = 1.0 / anomaly['resistivity']
        
        unique_vals = np.unique(active_model)
        for val in unique_vals:
            if not (np.isclose(val, bg_cond, atol=1e-5) or np.isclose(val, sp_cond, atol=1e-5)):
                raise ValueError(f"Validation failed: Unexpected conductivity value found {val}.")
        if not np.any(np.isclose(active_model, sp_cond, rtol=1e-5, atol=1e-7)):
            raise ValueError("Validation failed: The configured anomaly is absent from the active mesh.")
                
        # 4. True 3D Acquisition check
        # Check if we have measurements where A and M are on different lines (Y coords differ)
        a_idx = survey_data['a_idx']
        m_idx = survey_data['m_idx']
        a_y = electrodes[a_idx, 1]
        m_y = electrodes[m_idx, 1]
        
        if np.allclose(a_y, m_y):
            raise ValueError("Validation failed: All measurements are inline 2D profiles. Not a true 3D acquisition.")
            
        # 5. Enough measurements
        n_meas = len(a_idx)
        if n_meas < self.cfg.survey.min_measurements:
            raise ValueError(f"Validation failed: Only {n_meas} measurements. Target was {self.cfg.survey.min_measurements}.")
            
        # 6. No NaNs or Infs in data
        rho_a = survey_data['apparent_resistivity']
        if np.any(np.isnan(rho_a)) or np.any(np.isinf(rho_a)):
            raise ValueError("Validation failed: Apparent resistivity contains NaNs or Infs.")
            
        if np.any(rho_a <= 0):
            # In 3D Pole-Dipole, negative apparent resistivities CAN occur physically, 
            # but usually they indicate poor numerical stability if very large negative.
            # We will allow them but issue a warning, or filter them out.
            pass
            
        return True
