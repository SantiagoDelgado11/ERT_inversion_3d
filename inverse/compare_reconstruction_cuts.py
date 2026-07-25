import os
import sys
import yaml
import torch
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Añadir el root directory y la carpeta forward al path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
forward_dir = os.path.join(root_dir, 'forward')
sys.path.append(root_dir)
sys.path.append(forward_dir)

from mesh.mesh_generator import generate_mesh
from geology.conductivity_models import build_conductivity_model
from inverse.models import ConductivityNet

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Movernos al directorio forward para que todas las rutas relativas de los modulos funcionen
    os.chdir(forward_dir)
    
    # ---------------------------------------------------------
    # 1. GENERAR EL MODELO VERDADERO (FORWARD)
    # ---------------------------------------------------------
    print("Generando malla y modelo geologico verdadero...")
    mesh = generate_mesh()
    
    geology_cfg_path = "configs/geology.yaml"
    with open(geology_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # Configurar la esfera en (0, 0, -10) con radio 8.0 para que coincida
    config['geology']['background']['conductivity'] = 1.0 / 4500.0
    config['geology']['sphere']['conductivity']['min'] = 1.0 / 200.0
    config['geology']['sphere']['conductivity']['max'] = 1.0 / 200.0
    config['geology']['sphere']['radius']['min'] = 8.0
    config['geology']['sphere']['radius']['max'] = 8.0
    config['geology']['sphere']['center']['min_depth'] = 2.0
    config['geology']['sphere']['center']['max_depth'] = 18.0
    config['geology']['sphere']['center']['margin_x'] = 42.0
    config['geology']['sphere']['center']['margin_y'] = 12.0

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
    rho_core_true = rho_3d_full[pad_x:pad_x+nx, pad_y:pad_y+ny, pad_z:pad_z+nz]
    rho_core_true_blurred = scipy.ndimage.gaussian_filter(rho_core_true, sigma=2.0)

    core_x = mesh.cell_centers_x[pad_x:pad_x+nx]
    core_y = mesh.cell_centers_y[pad_y:pad_y+ny]
    core_z = mesh.cell_centers_z[pad_z:pad_z+nz]
    
    # ---------------------------------------------------------
    # 2. EVALUAR LA RECONSTRUCCIÓN (INVERSE PINN)
    # ---------------------------------------------------------
    print("Evaluando el modelo reconstruido (sigma_net)...")
    model_path = '../inverse/sigma_net.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado.")
        return
        
    sigma_net = ConductivityNet(hidden_layers=4, hidden_dim=128).to(device)
    sigma_net.load_state_dict(torch.load(model_path, map_location=device))
    sigma_net.eval()
    
    # Crear grid de evaluación solo sobre el core
    X, Y, Z = np.meshgrid(core_x, core_y, core_z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        sigma_pred_flat = sigma_net(coords_tensor).cpu().numpy().flatten()
    
    sigma_pred_grid = sigma_pred_flat.reshape((nx, ny, nz))
    rho_core_pred = 1.0 / sigma_pred_grid
    
    # ---------------------------------------------------------
    # 3. GRAFICAR COMPARACIÓN
    # ---------------------------------------------------------
    print("Generando graficos comparativos...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    
    z_val, y_val, x_val = -8.3, 1.0, 1.0
    z_idx = np.argmin(np.abs(core_z - z_val))
    y_idx = np.argmin(np.abs(core_y - y_val))
    x_idx = np.argmin(np.abs(core_x - x_val))
    
    XY_mesh_X, XY_mesh_Y = np.meshgrid(core_x, core_y)
    XZ_mesh_X, XZ_mesh_Z = np.meshgrid(core_x, core_z)
    YZ_mesh_Y, YZ_mesh_Z = np.meshgrid(core_y, core_z)
    
    def plot_row(row_axs, rho_data, title_prefix):
        # XY
        im1 = row_axs[0].pcolormesh(XY_mesh_X, XY_mesh_Y, rho_data[:, :, z_idx].T, cmap='viridis', shading='gouraud', vmin=200, vmax=3500)
        row_axs[0].set_title(f'{title_prefix} Horizontal (XY) Z={z_val}')
        row_axs[0].set_xlabel('X')
        row_axs[0].set_ylabel('Y')
        row_axs[0].set_xlim(-20, 20)
        row_axs[0].set_ylim(-20, 20)
        row_axs[0].set_aspect('equal')
        plt.colorbar(im1, ax=row_axs[0], label=r'$\Omega\cdot m$')
        
        # XZ
        im2 = row_axs[1].pcolormesh(XZ_mesh_X, XZ_mesh_Z, rho_data[:, y_idx, :].T, cmap='viridis', shading='gouraud', vmin=200, vmax=3500)
        row_axs[1].set_title(f'{title_prefix} Frontal (XZ) Y={y_val}')
        row_axs[1].set_xlabel('X')
        row_axs[1].set_ylabel('Z')
        row_axs[1].set_xlim(-20, 20)
        row_axs[1].set_ylim(-40, 0)
        row_axs[1].set_aspect('equal')
        plt.colorbar(im2, ax=row_axs[1], label=r'$\Omega\cdot m$')
        
        # YZ
        im3 = row_axs[2].pcolormesh(YZ_mesh_Y, YZ_mesh_Z, rho_data[x_idx, :, :].T, cmap='viridis', shading='gouraud', vmin=200, vmax=3500)
        row_axs[2].set_title(f'{title_prefix} Lateral (YZ) X={x_val}')
        row_axs[2].set_xlabel('Y')
        row_axs[2].set_ylabel('Z')
        row_axs[2].set_xlim(-20, 20)
        row_axs[2].set_ylim(-40, 0)
        row_axs[2].set_aspect('equal')
        plt.colorbar(im3, ax=row_axs[2], label=r'$\Omega\cdot m$')

    plot_row(axs[0], rho_core_true_blurred, "Verdadero -")
    plot_row(axs[1], rho_core_pred, "Reconstruido -")
    
    plt.tight_layout()
    plt.savefig('comparacion_resistividad.png', dpi=150)
    print("Guardado en 'comparacion_resistividad.png'")

if __name__ == '__main__':
    main()
