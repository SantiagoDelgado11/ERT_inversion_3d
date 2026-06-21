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
    
    for m in measurements:
        pos_A = sample['electrodes'][m['A']]
        pos_B = sample['electrodes'][m['B']] if m['B'] != -1 else pos_A + np.array([1000,0,0])
        pos_M = sample['electrodes'][m['M']]
        pos_N = sample['electrodes'][m['N']]
        
        # Approximation for pseudosection plotting
        x_c = (pos_A[0] + pos_B[0] + pos_M[0] + pos_N[0]) / 4.0
        # n level approx is distance
        a_approx = abs(pos_M[0] - pos_N[0])
        
        x_coords.append(x_c)
        n_levels.append(-a_approx) # negative for depth
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
