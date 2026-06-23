import torch
import matplotlib.pyplot as plt
import numpy as np

def verify_weight_mask():
    epsilon = 0.5
    R_scale = 3.0 * epsilon

    # Create a 2D grid for XZ plane (Y=0)
    x = np.linspace(-15, 15, 300)
    z = np.linspace(-15, 0, 150)
    X, Z = np.meshgrid(x, z)
    
    # Flatten grid and add Y=0
    coords_np = np.column_stack([X.ravel(), np.zeros_like(X.ravel()), Z.ravel()])
    coords = torch.tensor(coords_np, dtype=torch.float32)

    # Electrodes positions
    r_A = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    r_B = torch.tensor([[-10.0, 0.0, 0.0]], dtype=torch.float32)

    # Calculate distances
    dist_A = torch.sqrt(torch.sum((coords - r_A)**2, dim=1, keepdim=True) + 1e-12)
    dist_B = torch.sqrt(torch.sum((coords - r_B)**2, dim=1, keepdim=True) + 1e-12)

    # Tanh mask
    w_A = torch.tanh(dist_A / R_scale)
    w_B = torch.tanh(dist_B / R_scale)
    w_x = w_A * w_B

    # Reshape back to grid
    W_x_grid = w_x.numpy().reshape(X.shape)

    # Plot
    plt.figure(figsize=(10, 5))
    contour = plt.contourf(X, Z, W_x_grid, levels=50, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(contour, label='w(x) value')
    
    # Mark electrodes
    plt.scatter([10, -10], [0, 0], color='red', marker='v', s=100, label='Electrodes A/B')
    
    plt.title(r'Verification of Spatial Attenuation Weight $w(\mathbf{x})$ in XZ Plane (Y=0)')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.legend()
    plt.grid(alpha=0.3)
    
    out_file = 'mask_verification_XZ.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f'Verification plot saved to {out_file}')

if __name__ == '__main__':
    verify_weight_mask()
