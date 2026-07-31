import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]
    h5_filepath = repo_root / "dataset_output" / "campaign.h5"
    if not h5_filepath.exists():
        print(f"Error: Dataset {h5_filepath} no encontrado.")
        return
        
    with h5py.File(h5_filepath, 'r') as f:
        meta = dict(f['metadata'].attrs)
        c = f['ground_truth_conductivity']
        x = c['grid_x'][:]
        y = c['grid_y'][:]
        z = c['grid_z'][:]

    bg_rho = meta.get('bg_resistivity', 100.0)
    sph_r = meta.get('sphere_r', 10.0)
    sph_rho = meta.get('sphere_rho', 30.0)
    sph_x = meta.get('sphere_x', 50.0)
    sph_y = meta.get('sphere_y', 50.0)
    sph_z = meta.get('sphere_z', 25.0)

    # Reconstruimos la esfera analítica perfecta de Ground Truth
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist = np.sqrt((X - sph_x)**2 + (Y - sph_y)**2 + (Z - sph_z)**2)
    rho_3d = np.ones_like(dist) * bg_rho
    rho_3d[dist <= sph_r] = sph_rho

    # Búsqueda del CENTRO de la anomalía
    # En lugar de np.argmin (que da un borde del círculo), calculamos el centro de masa de la región anómala
    min_rho = np.min(rho_3d)
    anomaly_indices = np.where(rho_3d == min_rho)
    
    idx_x = int(np.round(np.mean(anomaly_indices[0])))
    idx_y = int(np.round(np.mean(anomaly_indices[1])))
    idx_z = int(np.round(np.mean(anomaly_indices[2])))

    target_x = x[idx_x]
    target_y = y[idx_y]
    target_z = z[idx_z]

    print("\n" + "="*50)
    print(f"*** CENTRO DE LA ANOMALÍA DETECTADO (GROUND TRUTH) ***")
    print(f"Coordenadas de los cortes: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
    print(f"Valor en ese punto: {rho_3d[idx_x, idx_y, idx_z]:.2f} Ohm-m")
    print("="*50 + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    
    # Corte Horizontal (XY)
    im0 = axes[0].imshow(rho_3d[:, :, idx_z].T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[0].set_title(f'Corte Horizontal (XY) Z={z[idx_z]:.2f}')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    fig.colorbar(im0, ax=axes[0], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Frontal (XZ)
    im1 = axes[1].imshow(rho_3d[:, idx_y, :].T, origin='upper', extent=[x_min, x_max, z_max, z_min], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[1].set_title(f'Corte Frontal (XZ) Y={y[idx_y]:.2f}')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    fig.colorbar(im1, ax=axes[1], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Lateral (YZ)
    im2 = axes[2].imshow(rho_3d[idx_x, :, :].T, origin='upper', extent=[y_min, y_max, z_max, z_min], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='equal')
    axes[2].set_title(f'Corte Lateral (YZ) X={x[idx_x]:.2f}')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    fig.colorbar(im2, ax=axes[2], label=r'Resistividad ($\Omega\cdot m$)')
    
    plt.tight_layout()
    out_file = Path(__file__).resolve().parent / 'ground_truth_result.png'
    plt.savefig(out_file, dpi=300)
    print(f"Imagen guardada exitosamente en {out_file}")

if __name__ == '__main__':
    main()
