import sys
import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from networks import ConductivityNet

def main():
    print("=== Iniciando Evaluación Estricta de Reconstrucción ERT ===")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo de Inferencia: {device}")
    
    # ---------------------------------------------------------
    # 1. Alineación de Mallas (Grid Interpolation)
    # ---------------------------------------------------------
    print("\n1. Carga de Ground Truth desde el tensor 3D...")
    h5_filepath = '../forward/dataset/dataset_validation.h5'
    
    if not os.path.exists(h5_filepath):
        print(f"Error: No se encontró el dataset en {h5_filepath}.")
        return

    with h5py.File(h5_filepath, 'r') as f:
        # Extraemos el tensor sigma del archivo dataset (ground truth de conductividad)
        sigma_true_grid = f['labels/true_resistivity_3d'][0]
        print(f"Shape del ground truth 3D: {sigma_true_grid.shape}")

    # 1. Aplicar la transformación inversa: rho_true = 1 / sigma_true
    rho_true_grid = 1.0 / sigma_true_grid

    # Crear la malla de inferencia espacial de la PINN exacta al grid de validación
    nx, ny, nz = rho_true_grid.shape
    # Asumiendo celdas de 2x2x2 m y origen centrado en X,Y y Z=0 en la superficie
    x = np.linspace(-49, 49, nx)
    y = np.linspace(-19, 19, ny)
    z = np.linspace(-39, -1, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    eval_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # ---------------------------------------------------------
    # Inferencia de la PINN
    # ---------------------------------------------------------
    print("Realizando inferencia espacial de la PINN...")
    sigma_net = ConductivityNet().to(device)
    model_path = 'sigma_net.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} no encontrado.")
        return
        
    sigma_net.load_state_dict(torch.load(model_path, map_location=device))
    sigma_net.eval()
    
    coords_tensor = torch.tensor(eval_coords, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        sigma_pred_flat = sigma_net(coords_tensor).cpu().numpy().flatten()
    
    sigma_pred_grid = sigma_pred_flat.reshape((nx, ny, nz))
    rho_pred_grid = 1.0 / sigma_pred_grid
    
    # ---------------------------------------------------------
    # 2. Cálculo de la Norma del Error (L2 Relativa)
    # ---------------------------------------------------------
    print("\n2. Cálculo de la Norma del Error (L2 Relativa) sobre Resistividad...")
    diff_norm = np.linalg.norm(rho_pred_grid - rho_true_grid)
    true_norm = np.linalg.norm(rho_true_grid)
    l2_rel_error = diff_norm / true_norm
    print(f"Error L_2 Relativo de la Resistividad: {l2_rel_error:.6f}")

    # ---------------------------------------------------------
    # 3. Índice de Similitud Estructural (SSIM 3D)
    # ---------------------------------------------------------
    print("\n3. Cálculo del Índice de Similitud Estructural (SSIM 3D) sobre Resistividad...")
    data_range = np.max(rho_true_grid) - np.min(rho_true_grid)
    ssim_val = ssim(rho_true_grid, rho_pred_grid, data_range=data_range, channel_axis=None)
    print(f"SSIM 3D Volumétrico: {ssim_val:.6f}")

    # ---------------------------------------------------------
    # 4. Análisis de Magnitud (Perfiles Lineales 1D)
    # ---------------------------------------------------------
    print("\n4. Extracción de Perfiles Lineales 1D (Análisis de Magnitud)...")
    # Buscamos el centroide de la anomalía en el Ground Truth de resistividad
    background = np.median(rho_true_grid)
    diff_from_bg = np.abs(rho_true_grid - background)
    max_idx = np.unravel_index(np.argmax(diff_from_bg), diff_from_bg.shape)
    cx, cy, cz = max_idx
    
    print(f"Centroide de Anomalía detectado en índices: X={cx}, Y={cy}, Z={cz}")
    print(f"Coordenadas físicas: X={x[cx]:.2f}, Y={y[cy]:.2f}, Z={z[cz]:.2f}")

    fig_1d, ax_1d = plt.subplots(1, 3, figsize=(15, 4))
    
    # Determinar rango dinámico compartido para los perfiles 1D
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

    # Perfil en X
    ax_1d[0].plot(x, rho_true_grid[:, cy, cz], 'k--', label='True')
    ax_1d[0].plot(x, rho_pred_grid[:, cy, cz], 'r-', label='Pred')
    ax_1d[0].set_title(f'Perfil X (Y={y[cy]:.1f}, Z={z[cz]:.1f})')
    ax_1d[0].set_xlabel('X')
    ax_1d[0].set_ylabel(r'Resistividad ($\Omega\cdot m$)')
    ax_1d[0].set_ylim(shared_ylim)
    ax_1d[0].legend()
    ax_1d[0].grid(True, alpha=0.3)

    # Perfil en Y
    ax_1d[1].plot(y, rho_true_grid[cx, :, cz], 'k--', label='True')
    ax_1d[1].plot(y, rho_pred_grid[cx, :, cz], 'r-', label='Pred')
    ax_1d[1].set_title(f'Perfil Y (X={x[cx]:.1f}, Z={z[cz]:.1f})')
    ax_1d[1].set_xlabel('Y')
    ax_1d[1].set_ylim(shared_ylim)
    ax_1d[1].legend()
    ax_1d[1].grid(True, alpha=0.3)

    # Perfil en Z
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

    # ---------------------------------------------------------
    # 5. Mapas Topológicos Residuales (Resistividad)
    # ---------------------------------------------------------
    print("\n5. Generando Mapas Topológicos Residuales (Resistividad)...")
    
    # Error absoluto
    abs_err_rho = np.abs(rho_pred_grid - rho_true_grid)
    
    fig_err, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap_divergent = 'seismic'
    
    # Corte XY
    im0 = axes[0].imshow(abs_err_rho[:, :, cz].T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap_divergent)
    axes[0].set_title(f'Error Abs XY (Z = {z[cz]:.1f})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    fig_err.colorbar(im0, ax=axes[0], label=r'$|\rho_{pred} - \rho_{true}|$ ($\Omega\cdot m$)')
    
    # Corte XZ
    im1 = axes[1].imshow(abs_err_rho[:, cy, :].T, origin='lower', extent=[x.min(), x.max(), z.min(), z.max()], cmap=cmap_divergent, aspect='auto')
    axes[1].set_title(f'Error Abs XZ (Y = {y[cy]:.1f})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    fig_err.colorbar(im1, ax=axes[1], label=r'$|\rho_{pred} - \rho_{true}|$ ($\Omega\cdot m$)')
    
    # Corte YZ
    im2 = axes[2].imshow(abs_err_rho[cx, :, :].T, origin='lower', extent=[y.min(), y.max(), z.min(), z.max()], cmap=cmap_divergent, aspect='auto')
    axes[2].set_title(f'Error Abs YZ (X = {x[cx]:.1f})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    fig_err.colorbar(im2, ax=axes[2], label=r'$|\rho_{pred} - \rho_{true}|$ ($\Omega\cdot m$)')
    
    plt.tight_layout()
    plt.savefig('evaluation_residual_maps.png', dpi=300)
    print("Mapas residuales guardados en 'evaluation_residual_maps.png'")
    
    # ---------------------------------------------------------
    # 6. Gráficas True vs Prediction (Scatter Plots)
    # ---------------------------------------------------------
    print("\n6. Generando Gráficas True vs Prediction...")
    # Conductividad True vs Pred
    sigma_true_flat = sigma_true_grid.flatten()
    sigma_pred_flat = sigma_pred_grid.flatten()
    
    # Potencial True vs Pred
    from pytorch_dataset import ERTDataset
    from networks import PotentialNet
    
    # Cargar datos empíricos de validación para comparar potenciales
    try:
        dataset = ERTDataset(h5_filepath=h5_filepath, n_pde=100, n_bc_surf=100, n_bc_inf=100, n_flux=100)
        sample = dataset[0]  # Tomamos el primer sample, correspondiente al true_resistivity_3d
        
        pot_net = PotentialNet().to(device)
        pot_model_path = 'pot_net.pth'
        if os.path.exists(pot_model_path):
            pot_net.load_state_dict(torch.load(pot_model_path, map_location=device))
            pot_net.eval()
            
            r_m = sample['data']['r_m'].to(device)
            source = sample['data']['source'].to(device)
            u_star = sample['data']['u_star'].cpu().numpy().flatten()
            
            with torch.no_grad():
                u_pred = pot_net(r_m, source).cpu().numpy().flatten()
                
            fig_scatter, ax_scatter = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter de Conductividad
            ax_scatter[0].scatter(sigma_true_flat, sigma_pred_flat, alpha=0.1, s=1, c='blue')
            ax_scatter[0].plot([sigma_true_flat.min(), sigma_true_flat.max()], 
                               [sigma_true_flat.min(), sigma_true_flat.max()], 'k--', lw=2)
            ax_scatter[0].set_xlabel('Conductividad True (S/m)')
            ax_scatter[0].set_ylabel('Conductividad Predicted (S/m)')
            ax_scatter[0].set_title('True vs Prediction: Conductividad')
            ax_scatter[0].grid(True, alpha=0.3)
            
            # Scatter de Potencial
            ax_scatter[1].scatter(u_star, u_pred, alpha=0.5, s=10, c='red')
            ax_scatter[1].plot([u_star.min(), u_star.max()], 
                               [u_star.min(), u_star.max()], 'k--', lw=2)
            ax_scatter[1].set_xlabel('Potencial True (V)')
            ax_scatter[1].set_ylabel('Potencial Predicted (V)')
            ax_scatter[1].set_title('True vs Prediction: Potencial Eléctrico')
            ax_scatter[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('evaluation_true_vs_pred.png', dpi=300)
            print("Gráfica True vs Prediction guardada en 'evaluation_true_vs_pred.png'")
        else:
            print(f"No se encontró {pot_model_path} para evaluar potencial.")
            raise Exception("No model")
    except Exception as e:
        print("Graficando solo conductividad debido a error/falta de modelo de potencial:", e)
        fig_scatter, ax_scatter = plt.subplots(1, 1, figsize=(6, 5))
        ax_scatter.scatter(sigma_true_flat, sigma_pred_flat, alpha=0.1, s=1, c='blue')
        ax_scatter.plot([sigma_true_flat.min(), sigma_true_flat.max()], 
                           [sigma_true_flat.min(), sigma_true_flat.max()], 'k--', lw=2)
        ax_scatter.set_xlabel('Conductividad True (S/m)')
        ax_scatter.set_ylabel('Conductividad Predicted (S/m)')
        ax_scatter.set_title('True vs Prediction: Conductividad')
        ax_scatter.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('evaluation_true_vs_pred.png', dpi=300)
        print("Gráfica True vs Prediction (Conductividad) guardada en 'evaluation_true_vs_pred.png'")

    print("\n=== Evaluación Completada Exitosamente ===")

if __name__ == "__main__":
    main()
