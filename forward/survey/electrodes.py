import numpy as np

def generate_surface_electrode_grid(num_x, num_y, spacing_x, spacing_y=None, center_x=0.0, center_y=0.0):
    """
    Generates a 2D rectangular grid of electrodes on the surface (z=0) 
    for 3D electrical resistivity tomography (ERT).
    """
    # Si no se define un espaciado en Y, asume una cuadrícula cuadrada
    if spacing_y is None:
        spacing_y = spacing_x
        
    length_x = (num_x - 1) * spacing_x
    length_y = (num_y - 1) * spacing_y
    
    start_x = center_x - length_x / 2.0
    start_y = center_y - length_y / 2.0
    
    # 1. Vectores lineales para cada eje
    x_lin = np.linspace(start_x, start_x + length_x, num_x)
    y_lin = np.linspace(start_y, start_y + length_y, num_y)
    
    # 2. Creación de la malla bidimensional
    X, Y = np.meshgrid(x_lin, y_lin)
    
    # 3. Aplanamiento (flattening) para vectorizar los puntos
    x_coords = X.flatten()
    y_coords = Y.flatten()
    z_coords = np.zeros_like(x_coords) # Z se mantiene en 0 (superficie)
    
    # Retorna matriz de dimensiones (N, 3) donde N = num_x * num_y
    return np.vstack((x_coords, y_coords, z_coords)).T
