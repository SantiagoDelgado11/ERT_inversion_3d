import numpy as np
import yaml
from geology.anomalies import Sphere, Ellipsoid, Block

def load_geology_config(config_path="configs/geology.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['geology']

def generate_random_anomalies(config, extent_x, extent_y, extent_z):
    num_anomalies = np.random.randint(1, config['max_anomalies'] + 1)
    anomalies = []
    
    res_min, res_max = config['resistivity_range']
    
    for _ in range(num_anomalies):
        res = np.exp(np.random.uniform(np.log(res_min), np.log(res_max)))
        
        # Randomly choose type
        atype = np.random.choice(['sphere', 'ellipsoid', 'block'])
        
        cx = np.random.uniform(extent_x[0], extent_x[1])
        cy = np.random.uniform(extent_y[0], extent_y[1])
        cz = np.random.uniform(extent_z[0], extent_z[1])
        
        if atype == 'sphere':
            r_min, r_max = config['sphere']['radius_range']
            r = np.random.uniform(r_min, r_max)
            anomalies.append(Sphere(res, cx, cy, cz, r))
        
        elif atype == 'ellipsoid':
            rx_min, rx_max = config['ellipsoid']['rx_range']
            ry_min, ry_max = config['ellipsoid']['ry_range']
            rz_min, rz_max = config['ellipsoid']['rz_range']
            rx = np.random.uniform(rx_min, rx_max)
            ry = np.random.uniform(ry_min, ry_max)
            rz = np.random.uniform(rz_min, rz_max)
            anomalies.append(Ellipsoid(res, cx, cy, cz, rx, ry, rz))
            
        elif atype == 'block':
            dx_min, dx_max = config['block']['dx_range']
            dy_min, dy_max = config['block']['dy_range']
            dz_min, dz_max = config['block']['dz_range']
            dx = np.random.uniform(dx_min, dx_max)
            dy = np.random.uniform(dy_min, dy_max)
            dz = np.random.uniform(dz_min, dz_max)
            anomalies.append(Block(res, cx - dx/2, cx + dx/2, 
                                   cy - dy/2, cy + dy/2, 
                                   cz - dz/2, cz + dz/2))
            
    return anomalies

def build_conductivity_model(mesh, config=None):
    if config is None:
        config = load_geology_config()
        
    bg_res = config.get('background_resistivity', 100.0)
    if isinstance(bg_res, list) and len(bg_res) == 2:
        bg_res = np.exp(np.random.uniform(np.log(bg_res[0]), np.log(bg_res[1])))
        
    sigma = np.ones(mesh.nC) / bg_res
    
    # Define extent where anomalies can occur (avoid padding cells)
    # E.g. core mesh region
    hx_core = config.get('core_x', [-30, 30])
    hy_core = config.get('core_y', [-10, 10])
    hz_core = config.get('core_z', [-25, -2]) # Keep anomalies a bit below surface
    
    anomalies = generate_random_anomalies(config, hx_core, hy_core, hz_core)
    
    # Evaluate over cell centers
    X, Y, Z = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], mesh.cell_centers[:, 2]
    
    for anomaly in anomalies:
        mask = anomaly.get_mask(X, Y, Z)
        sigma[mask] = 1.0 / anomaly.resistivity
        
    return sigma, anomalies
