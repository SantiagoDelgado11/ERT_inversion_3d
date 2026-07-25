import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import yaml

from mesh.mesh_generator import generate_mesh
from geology.conductivity_models import build_conductivity_model

# 1. Utilizar el modelo forward (mesh y conductivity_models) implementados
print("Generando malla y modelo geologico utilizando el forward implementado...")
mesh = generate_mesh()

# Cargamos la configuracion para generar una esfera especifica (o usamos los valores del config)
# Modificaremos temporalmente los parametros en memoria para asegurar que salga una esfera visible en el centro
with open("configs/geology.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Forzar la esfera en el centro para que los cortes coincidan visualmente
config['geology']['background']['conductivity'] = 1.0 / 200.0  # 200 Ohm-m
config['geology']['sphere']['conductivity']['min'] = 1.0 / 4500.0
config['geology']['sphere']['conductivity']['max'] = 1.0 / 4500.0
config['geology']['sphere']['radius']['min'] = 8.0
config['geology']['sphere']['radius']['max'] = 8.0
# Ajustamos los limites para obligar a que aparezca exactamente en (0, 0, -10)
config['geology']['sphere']['center']['min_depth'] = 2.0
config['geology']['sphere']['center']['max_depth'] = 18.0
config['geology']['sphere']['center']['margin_x'] = 42.0
config['geology']['sphere']['center']['margin_y'] = 12.0

sigma, anomalies = build_conductivity_model(mesh, config['geology'])
rho_true = 1.0 / sigma

# Reshape to 3D grid 
# Nota: La malla completa incluye celdas de padding (pad_x, pad_y, pad_z).
# Extraeremos solo la region central (el "core" del modelo).
with open("configs/mesh.yaml", 'r') as f:
    mesh_cfg = yaml.safe_load(f)['mesh']

pad_x, nx = mesh_cfg['pad_x'], mesh_cfg['nx']
pad_y, ny = mesh_cfg['pad_y'], mesh_cfg['ny']
pad_z, nz = mesh_cfg['pad_z'], mesh_cfg['nz']

# Extraer el "core" de resistividad (sin padding)
rho_3d_full = rho_true.reshape(mesh.shape_cells, order='F')
rho_core = rho_3d_full[pad_x:pad_x+nx, pad_y:pad_y+ny, pad_z:pad_z+nz]

# Aplicar filtro gaussiano para simular la visualizacion difuminada
rho_blurred = scipy.ndimage.gaussian_filter(rho_core, sigma=2.0)

# Obtener coordenadas del "core" de la malla
core_x = mesh.cell_centers_x[pad_x:pad_x+nx]
core_y = mesh.cell_centers_y[pad_y:pad_y+ny]
core_z = mesh.cell_centers_z[pad_z:pad_z+nz]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# --- Plot 1: Corte Horizontal (XY) en Z = -8.3 ---
# Find closest Z index
z_val = -8.3
z_idx = np.argmin(np.abs(core_z - z_val))
# Use pcolormesh for XY with cell centers to match C array
X_mesh, Y_mesh = np.meshgrid(core_x, core_y)
im1 = axs[0].pcolormesh(X_mesh, Y_mesh, rho_blurred[:, :, z_idx].T, cmap='viridis', shading='gouraud')
axs[0].set_title(f'Corte Horizontal (XY) en Z = {z_val}')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xlim(-20, 20)
axs[0].set_ylim(-20, 20)
axs[0].set_aspect('equal')
cbar1 = plt.colorbar(im1, ax=axs[0])
cbar1.set_label(r'Resistividad ($\Omega \cdot m$)')

# --- Plot 2: Corte Frontal (XZ) en Y = 1.0 ---
y_val = 1.0
y_idx = np.argmin(np.abs(core_y - y_val))
X_mesh, Z_mesh = np.meshgrid(core_x, core_z)
im2 = axs[1].pcolormesh(X_mesh, Z_mesh, rho_blurred[:, y_idx, :].T, cmap='viridis', shading='gouraud')
axs[1].set_title(f'Corte Frontal (XZ) en Y = {y_val}')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Z')
axs[1].set_xlim(-20, 20)
axs[1].set_ylim(-40, 0)
axs[1].set_aspect('equal')
cbar2 = plt.colorbar(im2, ax=axs[1])
cbar2.set_label(r'Resistividad ($\Omega \cdot m$)')

# --- Plot 3: Corte Lateral (YZ) en X = 1.0 ---
x_val = 1.0
x_idx = np.argmin(np.abs(core_x - x_val))
Y_mesh, Z_mesh = np.meshgrid(core_y, core_z)
im3 = axs[2].pcolormesh(Y_mesh, Z_mesh, rho_blurred[x_idx, :, :].T, cmap='viridis', shading='gouraud')
axs[2].set_title(f'Corte Lateral (YZ) en X = {x_val}')
axs[2].set_xlabel('Y')
axs[2].set_ylabel('Z')
axs[2].set_xlim(-20, 20)
axs[2].set_ylim(-40, 0)
axs[2].set_aspect('equal')
cbar3 = plt.colorbar(im3, ax=axs[2])
cbar3.set_label(r'Resistividad ($\Omega \cdot m$)')

plt.tight_layout()
plt.savefig('esfera_cortes.png')
print("Saved 'esfera_cortes.png'.")
