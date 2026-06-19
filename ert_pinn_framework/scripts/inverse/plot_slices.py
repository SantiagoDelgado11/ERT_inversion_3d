import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def main():
    npz_path = "ert_pinn_framework/outputs/exp_inversion_anomalia_5000/inversion_predictions.npz"
    data = np.load(npz_path)
    points = data["points"]
    sigma = data["conductivity"].reshape(-1)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Corte Horizontal (XY) cerca de Z=0.5
    z_target = 0.5
    mask = np.abs(points[:, 2] - z_target) < 0.3
    if np.sum(mask) > 10:
        slice_points = points[mask][:, :2]
        slice_sigma = sigma[mask]
        
        triang = mtri.Triangulation(slice_points[:, 0], slice_points[:, 1])
        im1 = axs[0].tricontourf(triang, slice_sigma, levels=20, cmap='viridis')
        axs[0].set_title("Corte Horizontal (Visto desde arriba) a Z=0.5m")
        axs[0].set_xlabel("Eje X (metros)")
        axs[0].set_ylabel("Eje Y (metros)")
        fig.colorbar(im1, ax=axs[0], label='Conductividad')
    
    # Corte Vertical (XZ) cerca de Y=0.0
    y_target = 0.0
    mask2 = np.abs(points[:, 1] - y_target) < 0.3
    if np.sum(mask2) > 10:
        slice_points2 = points[mask2][:, [0, 2]]
        slice_sigma2 = sigma[mask2]
        
        triang2 = mtri.Triangulation(slice_points2[:, 0], slice_points2[:, 1])
        im2 = axs[1].tricontourf(triang2, slice_sigma2, levels=20, cmap='viridis')
        axs[1].set_ylim(axs[1].get_ylim()[::-1])  # Invertir eje Z
        axs[1].set_title("Corte Transversal (Visto de lado) en Y=0.0m")
        axs[1].set_xlabel("Eje X (metros)")
        axs[1].set_ylabel("Profundidad Z (metros)")
        fig.colorbar(im2, ax=axs[1], label='Conductividad')
        
    plt.tight_layout()
    
    out_path = "C:/Users/Doc/.gemini/antigravity-ide/brain/56760ba2-eb19-4bc4-9315-f9ec1e345313/inversion_slices.png"
    plt.savefig(out_path, dpi=150)
    print(f"Grafico guardado en: {out_path}")

if __name__ == "__main__":
    main()
