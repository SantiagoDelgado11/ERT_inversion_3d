import torch
import numpy as np
import matplotlib.pyplot as plt
from networks import ConductivityNet
import os

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 1. Cargar la red pre-entrenada
    sigma_net = ConductivityNet().to(device)
    model_path = 'sigma_net.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado. Asegúrate de haber completado el entrenamiento.")
        return
        
    sigma_net.load_state_dict(torch.load(model_path, map_location=device))
    sigma_net.eval()
    
    # 2. Crear la malla de evaluación
    nx, ny, nz = 50, 50, 25
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-50, 50, ny)
    z = np.linspace(-50, 0, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    # 3. Inferencia
    with torch.no_grad():
        sigma_pred = sigma_net(coords_tensor).cpu().numpy().flatten()
    
    # 4. Convertir a resistividad
    sigma_pred = np.clip(sigma_pred, 1e-5, None)
    rho_pred = 1.0 / sigma_pred
    rho_3d = rho_pred.reshape((nx, ny, nz))
    
    # 5. Graficar cortes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Corte Horizontal (XY)
    idx_z = int(nz * 0.8) # Cerca de Z=-10
    im0 = axes[0].imshow(rho_3d[:, :, idx_z].T, origin='lower', extent=[-50, 50, -50, 50], cmap='viridis')
    axes[0].set_title(f'Corte Horizontal (XY) en Z = {z[idx_z]:.1f}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig.colorbar(im0, ax=axes[0], label='Resistividad ($\Omega\cdot m$)')
    
    # Corte Vertical (XZ)
    idx_y = ny // 2 # Y=0
    im1 = axes[1].imshow(rho_3d[:, idx_y, :].T, origin='lower', extent=[-50, 50, -50, 0], cmap='viridis', aspect='auto')
    axes[1].set_title(f'Corte Frontal (XZ) en Y = {y[idx_y]:.1f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    fig.colorbar(im1, ax=axes[1], label='Resistividad ($\Omega\cdot m$)')
    
    # Corte Lateral (YZ)
    idx_x = nx // 2 # X=0
    im2 = axes[2].imshow(rho_3d[idx_x, :, :].T, origin='lower', extent=[-50, 50, -50, 0], cmap='viridis', aspect='auto')
    axes[2].set_title(f'Corte Lateral (YZ) en X = {x[idx_x]:.1f}')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    fig.colorbar(im2, ax=axes[2], label='Resistividad ($\Omega\cdot m$)')
    
    plt.tight_layout()
    out_file = 'inversion_result.png'
    plt.savefig(out_file, dpi=300)
    print(f"Imagen guardada exitosamente en {out_file}")

if __name__ == '__main__':
    main()
