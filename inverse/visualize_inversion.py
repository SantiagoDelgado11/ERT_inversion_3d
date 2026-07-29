import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from models import ConductivityNet, MeasurementEncoder
from pytorch_dataset import ERTDataset
from pathlib import Path
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualizar la inversión de ERT 3D")
    parser.add_argument("--use_checkpoint", action="store_true", help="Cargar pesos desde checkpoint.pth")
    args = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    repo_root = Path(__file__).resolve().parents[1]
    h5_filepath = repo_root / "forward" / "dataset.h5"
    if not h5_filepath.exists():
        h5_filepath = repo_root / "inverse" / "single_experiment_data.h5"
        
    dataset = ERTDataset(h5_filepath=h5_filepath, n_pde=10, n_bc_surf=10, n_bc_inf=10, n_flux=10)
    data_samples = dataset[0]["data"]
    
    _r_A = data_samples["source"][..., 0:3].to(device)
    _r_B = data_samples["source"][..., 3:6].to(device)
    _r_m = data_samples["r_m"].to(device)
    _r_n = data_samples["r_n"].to(device) if "r_n" in data_samples else torch.zeros_like(_r_m)
    _delta_v = data_samples.get("delta_v", data_samples["u_star"]).to(device)
    
    _delta_v_scaled = torch.sign(_delta_v) * torch.log1p(torch.abs(_delta_v))
    encoder_input = torch.cat([_r_A, _r_B, _r_m, _r_n, _delta_v_scaled], dim=-1).unsqueeze(0)
    
    encoder = MeasurementEncoder(in_features=13, hidden_dim=128, latent_dim=128).to(device)
    sigma_net = ConductivityNet(hidden_layers=5, hidden_dim=256, latent_dim=128).to(device)

    if args.use_checkpoint:
        checkpoint_path = 'checkpoint.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Error: {checkpoint_path} no encontrado.")
            return
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'encoder_state_dict' in checkpoint and checkpoint['encoder_state_dict'] is not None:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        sigma_net.load_state_dict(checkpoint['sigma_net_state_dict'])
        print(f"Pesos cargados desde {checkpoint_path}")
    else:
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device, weights_only=True))
        sigma_net.load_state_dict(torch.load('sigma_net.pth', map_location=device, weights_only=True))
        print("Pesos finales cargados.")

    encoder.eval()
    sigma_net.eval()
    
    with torch.no_grad():
        latent = encoder(encoder_input)

    nx, ny, nz = 60, 60, 60
    x = np.linspace(-30, 30, nx)
    y = np.linspace(-30, 30, ny)
    z = np.linspace(-60, 0, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        expanded_latent = latent.expand(coords_tensor.shape[0], -1)
        sigma_pred = sigma_net(coords_tensor, latent=expanded_latent).cpu().numpy().flatten()
    
    rho_pred = 1.0 / sigma_pred
    rho_3d = rho_pred.reshape((nx, ny, nz))
    
    # ====================================================================
    # BÚSQUEDA DINÁMICA DE LA ANOMALÍA (Centro de Masa / Mínimo)
    # ====================================================================
    # Buscamos el índice 3D donde la resistividad es más baja
    min_idx = np.argmin(rho_3d)
    idx_x, idx_y, idx_z = np.unravel_index(min_idx, rho_3d.shape)

    target_x = x[idx_x]
    target_y = y[idx_y]
    target_z = z[idx_z]
    min_rho = rho_3d[idx_x, idx_y, idx_z]

    print("\n" + "="*50)
    print(f"🎯 CENTRO DE LA ANOMALÍA DETECTADO 🎯")
    print(f"Coordenadas: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
    print(f"Valor más bajo de Resistividad: {min_rho:.2f} Ohm-m")
    print("="*50 + "\n")

    # ====================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Corte Horizontal (XY) usando el índice dinámico
    im0 = axes[0].imshow(rho_3d[:, :, idx_z].T, origin='lower', extent=[-30, 30, -30, 30], cmap='jet', norm=LogNorm(vmin=1, vmax=10000))
    axes[0].set_title(f'Corte Horizontal (XY) Z={z[idx_z]:.2f}')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    fig.colorbar(im0, ax=axes[0], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Frontal (XZ) usando el índice dinámico
    im1 = axes[1].imshow(rho_3d[:, idx_y, :].T, origin='lower', extent=[-30, 30, -60, 0], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='auto')
    axes[1].set_title(f'Corte Frontal (XZ) Y={y[idx_y]:.2f}')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    fig.colorbar(im1, ax=axes[1], label=r'Resistividad ($\Omega\cdot m$)')
    
    # Corte Lateral (YZ) usando el índice dinámico
    im2 = axes[2].imshow(rho_3d[idx_x, :, :].T, origin='lower', extent=[-30, 30, -60, 0], cmap='jet', norm=LogNorm(vmin=1, vmax=10000), aspect='auto')
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