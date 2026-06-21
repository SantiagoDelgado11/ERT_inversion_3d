import discretize
import yaml
import numpy as np
from pathlib import Path

def load_mesh_config(config_path="configs/mesh.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['mesh']

def generate_mesh(config=None):
    if config is None:
        config = load_mesh_config()
    
    hx = config['hx']
    hy = config['hy']
    hz = config['hz']
    
    nx = config['nx']
    ny = config['ny']
    nz = config['nz']
    
    pad_x = config['pad_x']
    pad_y = config['pad_y']
    pad_z = config['pad_z']
    
    exp = config['expansion_rate']
    
    # Create padding cell arrays
    hx_pad = [(hx, pad_x, -exp), (hx, nx), (hx, pad_x, exp)]
    hy_pad = [(hy, pad_y, -exp), (hy, ny), (hy, pad_y, exp)]
    # Surface is at z=0, padding goes downwards only for z
    # Actually we can pad upwards (air) if we need to model topography, but standard ERT 
    # half-space puts the top boundary at z=0 and applies Neumann BC.
    # discretize TensorMesh naturally puts origin at 0,0,0 if not specified.
    # We will pad down (negative z) and set origin so top is at z=0.
    hz_pad = [(hz, pad_z, -exp), (hz, nz)]
    
    mesh = discretize.TensorMesh([hx_pad, hy_pad, hz_pad])
    
    # Shift mesh so that the center of the survey is roughly at x=0, y=0
    # and the top is at z=0
    x0 = -mesh.h[0].sum() / 2.0
    y0 = -mesh.h[1].sum() / 2.0
    z0 = 0.0 # top of the mesh
    
    mesh.origin = np.array([x0, y0, -mesh.h[2].sum()])
    
    return mesh

if __name__ == "__main__":
    mesh = generate_mesh()
    print(f"Generated mesh with {mesh.nC} cells.")
    print(f"Bounds: X {mesh.nodes_x[0]:.1f} to {mesh.nodes_x[-1]:.1f}")
    print(f"Bounds: Z {mesh.nodes_z[0]:.1f} to {mesh.nodes_z[-1]:.1f}")
