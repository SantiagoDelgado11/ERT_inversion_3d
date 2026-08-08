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
    # Use the exact discretized ground truth saved with the campaign.
    with h5py.File(h5_path, "r") as handle:
        gt = handle["ground_truth_conductivity"]
        sigma_true = np.asarray(gt["conductivity_tensor"], dtype=np.float32)
        gx = np.asarray(gt["grid_x"], dtype=np.float32)
        gy = np.asarray(gt["grid_y"], dtype=np.float32)
        gz = np.asarray(gt["grid_z"], dtype=np.float32)
        nx, ny, nz = sigma_true.shape
        target_rho = 1.0 / np.maximum(sigma_true, 1e-8)

    if rho_pred.shape != target_rho.shape:
        from scipy.ndimage import zoom
        factors = tuple(t / p for t, p in zip(target_rho.shape, rho_pred.shape))
        rho_pred = zoom(rho_pred, factors, order=1)
        
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
