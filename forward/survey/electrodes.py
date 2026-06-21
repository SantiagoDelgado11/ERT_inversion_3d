import numpy as np

def generate_surface_electrodes(num_electrodes, spacing, center_x=0.0, y=0.0):
    """
    Generates a straight line of electrodes on the surface (z=0).
    """
    length = (num_electrodes - 1) * spacing
    start_x = center_x - length / 2.0
    
    x = np.linspace(start_x, start_x + length, num_electrodes)
    y_coords = np.full_like(x, y)
    z_coords = np.zeros_like(x)
    
    return np.vstack((x, y_coords, z_coords)).T
