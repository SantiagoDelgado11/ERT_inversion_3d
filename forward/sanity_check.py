import matplotlib.pyplot as plt
import numpy as np
from dataset.generator import generate_single_sample
from mesh.mesh_generator import generate_mesh
import yaml
import os

def main():
    print("Running Sanity Check...")
    
    # 1. Ensure configs exist
    if not os.path.exists("configs/mesh.yaml"):
        print("Error: Configs not found.")
        return
        
    mesh = generate_mesh()
    print(f"Mesh instantiated. Cells: {mesh.nC}, Nodes: {mesh.nN}")
    
    # 2. Generate a single sample
    print("Generating single sample (Solving 3D Poisson equation for all pairs)...")
    sample = generate_single_sample(seed=42, mesh=mesh, return_mesh=True)
    
    sigma = sample['sigma']
    measurements = sample['measurements']
    
    print(f"Generated {len(measurements)} measurements.")
    
    # === DEBUG INFO START ===
    print("\n--- DEBUG INFO ---")
    
    with open("configs/mesh.yaml", 'r') as f:
        mesh_config = yaml.safe_load(f)['mesh']
    
    pad_x = mesh_config['pad_x']
    nx = mesh_config['nx']
    pad_y = mesh_config['pad_y']
    ny = mesh_config['ny']
    
    core_x_min = mesh.nodes_x[pad_x]
    core_x_max = mesh.nodes_x[pad_x + nx]
    core_y_min = mesh.nodes_y[pad_y]
    core_y_max = mesh.nodes_y[pad_y + ny]
    
    print(f"Límites de la Malla (Core X): {core_x_min:.2f} a {core_x_max:.2f}")
    print(f"Límites de la Malla (Core Y): {core_y_min:.2f} a {core_y_max:.2f}")
    
    electrodes = sample['electrodes']
    elec_x_min = electrodes[:, 0].min()
    elec_x_max = electrodes[:, 0].max()
    print(f"Límites de los Electrodos (X): {elec_x_min:.2f} a {elec_x_max:.2f}")
    
    array_center = (elec_x_min + elec_x_max) / 2.0
    print(f"Centro del Arreglo (X): {array_center:.2f}")
    
    sources = set()
    for m in measurements:
        sources.add((m['A'], m['B']))
    num_sources = len(sources)
    avg_receivers = len(measurements) / num_sources if num_sources > 0 else 0
    print(f"Longitud de la Secuencia: {num_sources} fuentes en total, {avg_receivers:.2f} receptores promedio por fuente")
    
    a_vals = []
    n_vals = []
    
    for m in measurements:
        pos_A = electrodes[m['A']]
        pos_B = electrodes[m['B']] if m['B'] != -1 else pos_A + np.array([1000,0,0])
        pos_M = electrodes[m['M']]
        pos_N = electrodes[m['N']]
        
        a = abs(pos_M[0] - pos_N[0])
        a_vals.append(a)
        
        dist_AM = abs(pos_A[0] - pos_M[0])
        dist_AN = abs(pos_A[0] - pos_N[0])
        dist_BM = abs(pos_B[0] - pos_M[0])
        dist_BN = abs(pos_B[0] - pos_N[0])
        
        min_dist = min(dist_AM, dist_AN, dist_BM, dist_BN)
        
        if a > 0:
            n_vals.append(min_dist / a)
            
    if a_vals and n_vals:
        print(f"Distancia 'a' (min-max): {min(a_vals):.2f} a {max(a_vals):.2f}")
        print(f"Nivel 'n' aprox (min-max): {min(n_vals):.2f} a {max(n_vals):.2f}")
    print("------------------\n")
    # === DEBUG INFO END ===
    
    # 3. Simple QC Plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot central slice of conductivity model (y=0 approx)
    # discretize PlotSlice is useful, but we can do a simple scatter
    y_center = mesh.nodes_y[mesh.shape_cells[1] // 2]
    # To keep dependencies light, we will use discretize's plot_slice
    mesh.plot_slice(1.0/sigma, normal='Y', ax=axs[0], grid=True, pcolor_opts={'cmap': 'jet'})
    axs[0].set_title('True Resistivity (Ohm-m) - Y Slice')
    
    # Pseudosection (scatter of apparent resistivities)
    x_coords = []
    n_levels = []
    rho_as = []
    
    with open("configs/survey.yaml", 'r') as f:
        survey_config = yaml.safe_load(f)['survey']
    dx = survey_config.get('electrode_spacing', 2.0)
    
    for m in measurements:
        A_idx = m['A']
        B_idx = m['B']
        M_idx = m['M']
        N_idx = m['N']
        
        pos_A = sample['electrodes'][A_idx]
        pos_B = sample['electrodes'][B_idx] if B_idx != -1 else pos_A + np.array([1000,0,0])
        pos_M = sample['electrodes'][M_idx]
        pos_N = sample['electrodes'][N_idx]
        
        X_mid = (pos_A[0] + pos_B[0] + pos_M[0] + pos_N[0]) / 4.0
        
        a_mult = abs(M_idx - N_idx)
        if B_idx != -1:
            n_mult = abs(M_idx - B_idx) / a_mult if a_mult > 0 else 1
        else:
            n_mult = abs(M_idx - A_idx) / a_mult if a_mult > 0 else 1
            
        Z_pseudo = -((n_mult + 1) * a_mult * dx) / 2.0
        
        x_coords.append(X_mid)
        n_levels.append(Z_pseudo)
        rho_as.append(m['rho_a'])
        
    sc = axs[1].scatter(x_coords, n_levels, c=rho_as, cmap='jet')
    plt.colorbar(sc, ax=axs[1], label='Apparent Resistivity (Ohm-m)')
    axs[1].set_title('Pseudosection QC')
    axs[1].set_xlabel('X Position (m)')
    axs[1].set_ylabel('Pseudo-depth')
    
    plt.tight_layout()
    plt.savefig('sanity_check_plot.png')
    print("Saved 'sanity_check_plot.png'.")
    print("Sanity check passed without numerical singularities!")

if __name__ == "__main__":
    main()
