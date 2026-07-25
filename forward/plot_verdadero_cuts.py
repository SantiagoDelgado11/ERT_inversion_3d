import os
import sys
import yaml
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Añadir el root directory y la carpeta forward al path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
forward_dir = os.path.join(root_dir, 'forward')
sys.path.append(root_dir)
sys.path.append(forward_dir)

from mesh.mesh_generator import generate_mesh
from geology.conductivity_models import build_conductivity_model

def main():
    os.chdir(forward_dir)
    print("Generando malla y modelo geologico verdadero...")
    mesh = generate_mesh()
    
    geology_cfg_path = "configs/geology.yaml"
    with open(geology_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # Utilizamos la configuración generada directamente desde el yaml
    sigma_true, anomalies = build_conductivity_model(mesh, config['geology'])
    rho_true = 1.0 / sigma_true
    
    # Extraer el core de la malla sin padding
    mesh_cfg_path = "configs/mesh.yaml"
    with open(mesh_cfg_path, 'r') as f:
        mesh_cfg = yaml.safe_load(f)['mesh']

    pad_x, nx = mesh_cfg['pad_x'], mesh_cfg['nx']
    pad_y, ny = mesh_cfg['pad_y'], mesh_cfg['ny']
    pad_z, nz = mesh_cfg['pad_z'], mesh_cfg['nz']

    rho_3d_full = rho_true.reshape(mesh.shape_cells, order='F')
    rho_full_blurred = scipy.ndimage.gaussian_filter(rho_3d_full, sigma=2.0)

    full_x = mesh.cell_centers_x
    full_y = mesh.cell_centers_y
    full_z = mesh.cell_centers_z
    
    print("Generando graficos comparativos...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    z_val, y_val, x_val = -25.0, 0.0, 0.0
    z_idx = np.argmin(np.abs(full_z - z_val))
    y_idx = np.argmin(np.abs(full_y - y_val))
    x_idx = np.argmin(np.abs(full_x - x_val))
    
    XY_mesh_X, XY_mesh_Y = np.meshgrid(full_x, full_y)
    XZ_mesh_X, XZ_mesh_Z = np.meshgrid(full_x, full_z)
    YZ_mesh_Y, YZ_mesh_Z = np.meshgrid(full_y, full_z)
    
    def plot_row(row_axs, rho_data, title_prefix):
        norm = mcolors.LogNorm(vmin=1, vmax=10000)
        cmap = 'jet'
        
        # XY
        im1 = row_axs[0].pcolormesh(XY_mesh_X, XY_mesh_Y, rho_data[:, :, z_idx].T, cmap=cmap, shading='gouraud', norm=norm)
        row_axs[0].set_title(f'Corte Horizontal (XY) Z={z_val}')
        row_axs[0].set_xlabel('X (m)')
        row_axs[0].set_ylabel('Y (m)')
        row_axs[0].set_xlim(-30, 30)
        row_axs[0].set_ylim(-30, 30)
        row_axs[0].set_aspect('equal')
        plt.colorbar(im1, ax=row_axs[0], label=r'Resistividad ($\Omega\cdot m$)')
        
        # XZ
        im2 = row_axs[1].pcolormesh(XZ_mesh_X, XZ_mesh_Z, rho_data[:, y_idx, :].T, cmap=cmap, shading='gouraud', norm=norm)
        row_axs[1].set_title(f'Corte Frontal (XZ) Y={y_val}')
        row_axs[1].set_xlabel('X (m)')
        row_axs[1].set_ylabel('Z (m)')
        row_axs[1].set_xlim(-30, 30)
        row_axs[1].set_ylim(-60, 0)
        row_axs[1].set_aspect('equal')
        plt.colorbar(im2, ax=row_axs[1], label=r'Resistividad ($\Omega\cdot m$)')
        
        # YZ
        im3 = row_axs[2].pcolormesh(YZ_mesh_Y, YZ_mesh_Z, rho_data[x_idx, :, :].T, cmap=cmap, shading='gouraud', norm=norm)
        row_axs[2].set_title(f'Corte Lateral (YZ) X={x_val}')
        row_axs[2].set_xlabel('Y (m)')
        row_axs[2].set_ylabel('Z (m)')
        row_axs[2].set_xlim(-30, 30)
        row_axs[2].set_ylim(-60, 0)
        row_axs[2].set_aspect('equal')
        plt.colorbar(im3, ax=row_axs[2], label=r'Resistividad ($\Omega\cdot m$)')

    plot_row(axs, rho_full_blurred, "Verdadero -")
    
    plt.tight_layout()
    output_path = 'esfera_cortes_verdadero.png'
    plt.savefig(output_path, dpi=150)
    print(f"Guardado en '{output_path}'")
    
    # Copiar a artifacts para que Antigravity lo pueda visualizar
    import shutil
    artifacts_dir = r"C:\Users\Doc\.gemini\antigravity-ide\brain\94509920-1d43-4fa8-afe0-2187c478cc38"
    shutil.copy(output_path, os.path.join(artifacts_dir, 'esfera_cortes_verdadero.png'))

if __name__ == '__main__':
    main()
