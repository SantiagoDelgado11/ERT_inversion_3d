import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from models import ConductivityNet
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualizar Inversión ERT 3D")
    parser.add_argument("--use_checkpoint", action="store_true", help="Usar el checkpoint más reciente en checkpoints/latest_checkpoint.pth")
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 1. Cargar la red pre-entrenada
    sigma_net = ConductivityNet(hidden_layers=4, hidden_dim=128).to(device)
    
    if args.use_checkpoint:
        model_path = 'checkpoints/latest_checkpoint.pth'
        if not os.path.exists(model_path):
            print(f"Error: {model_path} no encontrado.")
            return
        checkpoint = torch.load(model_path, map_location=device)
        if 'm_net_state_dict' in checkpoint:
            sigma_net.load_state_dict(checkpoint['m_net_state_dict'])
            print(f"Cargado estado m_net_state_dict desde checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")
        elif 'sigma_net_state_dict' in checkpoint:
            sigma_net.load_state_dict(checkpoint['sigma_net_state_dict'])
            print(f"Cargado estado sigma_net_state_dict desde checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            print("No se encontró el estado de la red de conductividad en el checkpoint.")
            return
    else:
        model_path = 'sigma_net.pth'
        if not os.path.exists(model_path):
            print(f"Error: {model_path} no encontrado. Asegúrate de haber completado el entrenamiento.")
            return
            
        sigma_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Cargado desde {model_path}")
        
    sigma_net.eval()
    
    # 2. Crear la malla de evaluación
    nx, ny, nz = 50, 50, 25
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    z = np.linspace(0, 50, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    # 3. Inferencia
    with torch.no_grad():
        sigma_pred = sigma_net(coords_tensor).cpu().numpy().flatten()
    
    # 4. Convertir a resistividad
    rho_pred = 1.0 / sigma_pred
    rho_3d = rho_pred.reshape((nx, ny, nz))
    
    # 5. Graficar cortes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Corte Horizontal (XY)
    idx_z = nz // 2 # Z=25
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    im0 = axes[0].imshow(rho_3d[:, :, idx_z].T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[0].set_title(f'Corte Horizontal (XY) Z={z[idx_z]:.2f}')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    fig.colorbar(im0, ax=axes[0], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Vertical (XZ)
    idx_y = ny // 2
    im1 = axes[1].imshow(rho_3d[:, idx_y, :].T, origin='upper', extent=[x_min, x_max, z_max, z_min], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[1].set_title(f'Corte Frontal (XZ) Y={y[idx_y]:.2f}')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    fig.colorbar(im1, ax=axes[1], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Lateral (YZ)
    idx_x = nx // 2
    im2 = axes[2].imshow(rho_3d[idx_x, :, :].T, origin='upper', extent=[y_min, y_max, z_max, z_min], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[2].set_title(f'Corte Lateral (YZ) X={x[idx_x]:.2f}')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    fig.colorbar(im2, ax=axes[2], label=r'Resistividad ($\Omega\cdot m$)')
    
    plt.tight_layout()
    out_file = 'inversion_result.png'
    plt.savefig(out_file, dpi=300)
    print(f"Imagen guardada exitosamente en {out_file}")
    plt.show()

if __name__ == '__main__':
    main()
