import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

def plot_comparison(npy_path, h5_path, output_png):
    # Cargar predicción
    rho_pred = np.load(npy_path)
    if rho_pred.shape == (50, 50, 25):
        rho_pred = np.repeat(np.repeat(np.repeat(rho_pred, 2, 0), 2, 1), 2, 2)
        
    # Reconstruir Ground Truth
    with h5py.File(h5_path, "r") as handle:
        metadata = handle["metadata"].attrs
        nx, ny, nz = 100, 100, 50
        gx = np.linspace(0.5, 99.5, nx)
        gy = np.linspace(0.5, 99.5, ny)
        gz = np.linspace(0.5, 49.5, nz)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        
        sx = float(metadata.get("sphere_x", 50.0))
        sy = float(metadata.get("sphere_y", 50.0))
        sz = float(metadata.get("sphere_z", 25.0))
        sr = float(metadata.get("sphere_r", 10.0))
        
        distance = np.sqrt((X - sx)**2 + (Y - sy)**2 + (Z - sz)**2)
        target_rho = np.full((nx, ny, nz), float(metadata.get("bg_resistivity", 100.0)), dtype=np.float32)
        target_rho[distance <= sr] = float(metadata.get("sphere_rho", 30.0))
        
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # Limites comunes para la escala de color (igualados a deep_inversion.py)
    vmin, vmax = 1, 10000 
    
    # Fila 1: Ground Truth
    axes[0, 0].imshow(target_rho[:, :, nz//2].T, origin="lower", extent=[0, 100, 0, 100], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[0, 0].set_title("Verdad Terreno - Vista Planta (XY, Z=25m)")
    axes[0, 0].set_ylabel("Y (m)")
    
    axes[0, 1].imshow(target_rho[:, ny//2, :].T, origin="upper", extent=[0, 100, 50, 0], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[0, 1].set_title("Verdad Terreno - Perfil Lateral (XZ, Y=50m)")
    axes[0, 1].set_ylabel("Profundidad (m)")
    
    axes[0, 2].imshow(target_rho[nx//2, :, :].T, origin="upper", extent=[0, 100, 50, 0], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[0, 2].set_title("Verdad Terreno - Perfil Frontal (YZ, X=50m)")
    
    # Fila 2: Predicción
    im = axes[1, 0].imshow(rho_pred[:, :, nz//2].T, origin="lower", extent=[0, 100, 0, 100], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[1, 0].set_title(f"Predicción - Vista Planta (XY)")
    axes[1, 0].set_ylabel("Y (m)")
    axes[1, 0].set_xlabel("X (m)")
    
    axes[1, 1].imshow(rho_pred[:, ny//2, :].T, origin="upper", extent=[0, 100, 50, 0], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[1, 1].set_title(f"Predicción - Perfil Lateral (XZ)")
    axes[1, 1].set_ylabel("Profundidad (m)")
    axes[1, 1].set_xlabel("X (m)")
    
    axes[1, 2].imshow(rho_pred[nx//2, :, :].T, origin="upper", extent=[0, 100, 50, 0], cmap="jet", norm=LogNorm(vmin, vmax))
    axes[1, 2].set_title(f"Predicción - Perfil Frontal (YZ)")
    axes[1, 2].set_xlabel("Y (m)")
    
    # Barra de color global
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=r"Resistividad ($\Omega\cdot m$)")
    
    plt.subplots_adjust(right=0.9, hspace=0.3)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"¡Imagen comparativa guardada en {output_png}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Archivo .npy de la red")
    parser.add_argument("--h5", default="dataset_output/campaign.h5")
    parser.add_argument("--out", required=True, help="Ruta de la imagen .png de salida")
    args = parser.parse_args()
    
    plot_comparison(args.npy, args.h5, args.out)
