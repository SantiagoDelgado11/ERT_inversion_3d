import numpy as np
import yaml
from geology.anomalies import SphereAnomaly

def load_geology_config(config_path="configs/geology.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['geology']

def generate_single_sphere(config, extent_x, extent_y, extent_z):
    """
    Genera exactamente una única anomalía esférica asegurando que
    esté completamente dentro del dominio definido por config.
    """
    sphere_cfg = config['sphere']
    
    r_min = sphere_cfg['radius']['min']
    r_max = sphere_cfg['radius']['max']
    center_cfg = sphere_cfg['center']
    margin_x = center_cfg['margin_x']
    margin_y = center_cfg['margin_y']
    min_depth = center_cfg['min_depth']
    max_depth = center_cfg['max_depth']

    # Limitar r_max basado en los márgenes y el tamaño del dominio
    max_allowed_r_x = (extent_x[1] - extent_x[0] - 2 * margin_x) / 2.0
    max_allowed_r_y = (extent_y[1] - extent_y[0] - 2 * margin_y) / 2.0
    max_allowed_r_z = (max_depth - min_depth) / 2.0
    
    actual_r_max = min(r_max, max_allowed_r_x, max_allowed_r_y, max_allowed_r_z)
    
    if r_min > actual_r_max:
        raise ValueError(f"No se puede colocar la esfera. r_min ({r_min}) es mayor que el radio máximo permitido ({actual_r_max}) dadas las dimensiones y los márgenes.")
        
    r = np.random.uniform(r_min, actual_r_max)
    
    cond_min = sphere_cfg['conductivity']['min']
    cond_max = sphere_cfg['conductivity']['max']
    # Log-uniform para la conductividad
    cond = np.exp(np.random.uniform(np.log(cond_min), np.log(cond_max)))
    
    # Z axis in SimPEG mesh is typically vertical, with max_z = 0 (surface) or close to it.
    surface_z = extent_z[1]
    
    # Calcular límites válidos para el centro para que la esfera no salga
    # del dominio y respete los márgenes
    valid_x_min = extent_x[0] + margin_x + r
    valid_x_max = extent_x[1] - margin_x - r
    valid_y_min = extent_y[0] + margin_y + r
    valid_y_max = extent_y[1] - margin_y - r
    
    valid_z_max = surface_z - min_depth - r
    valid_z_min = surface_z - max_depth + r
    
    # Check bounds
    if valid_x_min > valid_x_max or valid_y_min > valid_y_max or valid_z_min > valid_z_max:
        raise ValueError(f"No se puede colocar la esfera con los márgenes especificados. Los rangos son muy restrictivos o el radio ({r}) es muy grande.")
        
    cx = np.random.uniform(valid_x_min, valid_x_max)
    cy = np.random.uniform(valid_y_min, valid_y_max)
    cz = np.random.uniform(valid_z_min, valid_z_max)
    
    return [SphereAnomaly(cond, cx, cy, cz, r)]

def build_conductivity_model(mesh, config=None):
    if config is None:
        config = load_geology_config()
        
    bg_cond = config['background']['conductivity']
    sigma = np.ones(mesh.nC) * bg_cond
    
    # Define extent where anomalies can occur (avoid padding cells)
    with open("configs/mesh.yaml", 'r') as f:
        mesh_config = yaml.safe_load(f)['mesh']
    
    pad_x = mesh_config['pad_x']
    nx = mesh_config['nx']
    pad_y = mesh_config['pad_y']
    ny = mesh_config['ny']
    pad_z = mesh_config['pad_z']
    nz = mesh_config['nz']
    
    hx_core = [mesh.nodes_x[pad_x], mesh.nodes_x[pad_x + nx]]
    hy_core = [mesh.nodes_y[pad_y], mesh.nodes_y[pad_y + ny]]
    
    z_min = mesh.nodes_z[pad_z]
    z_max = mesh.nodes_z[pad_z + nz]
    hz_core = [z_min, z_max]
    
    anomalies = generate_single_sphere(config, hx_core, hy_core, hz_core)
    
    # Evaluate over cell centers
    X, Y, Z = mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], mesh.cell_centers[:, 2]
    
    for anomaly in anomalies:
        mask = anomaly.get_mask(X, Y, Z)
        sigma[mask] = anomaly.conductivity
        
    return sigma, anomalies
