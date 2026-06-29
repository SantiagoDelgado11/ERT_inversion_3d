import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from networks import ConductivityNet

def main():
    print("=== Generando Perfiles 1D de Inversión ===")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    h5_filepath = '../forward/dataset/dataset_validation.h5'
    if not os.path.exists(h5_filepath):
        print(f"Error: No se encontró el dataset en {h5_filepath}.")
        return

    with h5py.File(h5_filepath, 'r') as f:
        # Extraemos el tensor (en realidad es conductividad, a pesar del nombre en el h5)
        sigma_true_grid = f['labels/true_resistivity_3d'][0]

    # Convertir a resistividad
    rho_true_grid = 1.0 / sigma_true_grid

    # Crear la malla de evaluación
    nx, ny, nz = rho_true_grid.shape
    x = np.linspace(-49, 49, nx)
    y = np.linspace(-19, 19, ny)
    z = np.linspace(-39, -1, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    eval_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Cargar el modelo ConductivityNet
    sigma_net = ConductivityNet().to(device)
    model_path = 'sigma_net.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado.")
        return
        
    sigma_net.load_state_dict(torch.load(model_path, map_location=device))
    sigma_net.eval()
    
    coords_tensor = torch.tensor(eval_coords, dtype=torch.float32).to(device)
    
    # Inferencia
    with torch.no_grad():
        sigma_pred_flat = sigma_net(coords_tensor).cpu().numpy().flatten()
    
    sigma_pred_grid = sigma_pred_flat.reshape((nx, ny, nz))
    rho_pred_grid = 1.0 / sigma_pred_grid
    
    # Buscar centroide de la anomalía
    background = np.median(rho_true_grid)
    diff_from_bg = np.abs(rho_true_grid - background)
    max_idx = np.unravel_index(np.argmax(diff_from_bg), diff_from_bg.shape)
    cx, cy, cz = max_idx
    
    print(f"Anomalía centrada en índices X={cx}, Y={cy}, Z={cz}")
    
    fig_1d, ax_1d = plt.subplots(1, 3, figsize=(15, 4))
    
    # Limites en Y compartidos
    y_min = min(
        np.min(rho_true_grid[:, cy, cz]), np.min(rho_pred_grid[:, cy, cz]),
        np.min(rho_true_grid[cx, :, cz]), np.min(rho_pred_grid[cx, :, cz]),
        np.min(rho_true_grid[cx, cy, :]), np.min(rho_pred_grid[cx, cy, :])
    )
    y_max = max(
        np.max(rho_true_grid[:, cy, cz]), np.max(rho_pred_grid[:, cy, cz]),
        np.max(rho_true_grid[cx, :, cz]), np.max(rho_pred_grid[cx, :, cz]),
        np.max(rho_true_grid[cx, cy, :]), np.max(rho_pred_grid[cx, cy, :])
    )
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    shared_ylim = (y_min - margin, y_max + margin)

    # Perfil X
    ax_1d[0].plot(x, rho_true_grid[:, cy, cz], 'k--', label='True')
    ax_1d[0].plot(x, rho_pred_grid[:, cy, cz], 'r-', label='Pred')
    ax_1d[0].set_title(f'Perfil X (Y={y[cy]:.1f}, Z={z[cz]:.1f})')
    ax_1d[0].set_xlabel('X')
    ax_1d[0].set_ylabel(r'Resistividad ($\Omega\cdot m$)')
    ax_1d[0].set_ylim(shared_ylim)
    ax_1d[0].legend()
    ax_1d[0].grid(True, alpha=0.3)

    # Perfil Y
    ax_1d[1].plot(y, rho_true_grid[cx, :, cz], 'k--', label='True')
    ax_1d[1].plot(y, rho_pred_grid[cx, :, cz], 'r-', label='Pred')
    ax_1d[1].set_title(f'Perfil Y (X={x[cx]:.1f}, Z={z[cz]:.1f})')
    ax_1d[1].set_xlabel('Y')
    ax_1d[1].set_ylim(shared_ylim)
    ax_1d[1].legend()
    ax_1d[1].grid(True, alpha=0.3)

    # Perfil Z
    ax_1d[2].plot(z, rho_true_grid[cx, cy, :], 'k--', label='True')
    ax_1d[2].plot(z, rho_pred_grid[cx, cy, :], 'r-', label='Pred')
    ax_1d[2].set_title(f'Perfil Z (X={x[cx]:.1f}, Y={y[cy]:.1f})')
    ax_1d[2].set_xlabel('Z')
    ax_1d[2].set_ylim(shared_ylim)
    ax_1d[2].legend()
    ax_1d[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_1d_profiles.png', dpi=300)
    print("Grafico de perfiles 1D guardado en 'evaluation_1d_profiles.png'")

if __name__ == "__main__":
    main()
