import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sample_summary(output_dir: str, survey_data: dict, anomaly: dict, domain_cfg):
    """
    Creates a summary figure for the generated campaign to validate visually.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Electrode Distribution (Surface)
    ax = axs[0, 0]
    electrodes = survey_data['electrodes']
    ax.scatter(electrodes[:, 0], electrodes[:, 1], c='black', marker='.', s=10)
    ax.set_title("Electrode Grid (Z=0)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(0, domain_cfg.x_length)
    ax.set_ylim(0, domain_cfg.y_length)
    ax.set_aspect('equal')
    
    # Add sphere projection on surface
    circle = plt.Circle((anomaly['x'], anomaly['y']), anomaly['r'], color='red', fill=False, linestyle='--')
    ax.add_patch(circle)
    
    # 2. XZ Cross section of the sphere position
    ax = axs[0, 1]
    ax.set_title(f"XZ Cross Section at Y={anomaly['y']:.1f}")
    ax.set_xlim(0, domain_cfg.x_length)
    ax.set_ylim(domain_cfg.z_length, 0) # Depth points down
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth (Z) (m)")
    
    circle_xz = plt.Circle((anomaly['x'], anomaly['z']), anomaly['r'], color='red', fill=True, alpha=0.5)
    ax.add_patch(circle_xz)
    ax.axhline(0, color='black', linewidth=2) # Surface
    ax.set_aspect('equal')
    
    # 3. Apparent Resistivity pseudo-histogram
    ax = axs[1, 0]
    rho_a = survey_data['apparent_resistivity']
    # Filter extremes for better visualization
    rho_a_valid = rho_a[(rho_a > 0) & (rho_a < 1000)]
    ax.hist(rho_a_valid, bins=50, color='blue', alpha=0.7)
    ax.set_title("Apparent Resistivity Distribution")
    ax.set_xlabel("Apparent Resistivity (Ohm.m)")
    ax.set_ylabel("Frequency")
    ax.axvline(domain_cfg.bg_resistivity, color='gray', linestyle='dashed', label='Background')
    ax.axvline(anomaly['resistivity'], color='red', linestyle='dashed', label='Anomaly')
    ax.legend()
    
    # 4. Coverage (Log(Abs(K)) histogram)
    ax = axs[1, 1]
    k_factors = survey_data['k_factor']
    ax.hist(np.log10(np.abs(k_factors) + 1e-5), bins=50, color='green', alpha=0.7)
    ax.set_title("Geometric Factor Distribution (log10|K|)")
    ax.set_xlabel("log10(|K|)")
    ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    out_file = Path(output_dir) / "campaign_summary.png"
    plt.savefig(out_file)
    plt.close(fig)

def plot_3d_model_pyvista(output_dir: str, mesh, active_model, anomaly, domain_cfg):
    """
    Plots the 3D model using PyVista (if installed).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Skipping 3D visualization.")
        return
        
    try:
        # Convert discretize TreeMesh to PyVista UnstructuredGrid
        pv_mesh = mesh.to_vtk()
        
        # We need to assign the active model to the pv_mesh
        # pv_mesh has all cells. active_model is only for active cells.
        full_model = np.ones(mesh.n_cells) * (1.0 / domain_cfg.bg_resistivity)
        active_indices = mesh.cell_centers[:, 2] >= -1e-5
        full_model[active_indices] = active_model
        
        # In resistivity, we prefer plotting log10(resistivity)
        res_model = 1.0 / full_model
        pv_mesh.cell_data['Resistivity'] = np.log10(res_model)
        
        # Create a plotter
        plotter = pv.Plotter(off_screen=True)
        
        # Threshold to only show the anomaly
        thresholded = pv_mesh.threshold(np.log10(anomaly['resistivity']) + 0.1)
        plotter.add_mesh(thresholded, color='red', label='Anomaly')
        
        # Add a bounding box or outline
        plotter.add_mesh(pv_mesh.outline(), color='black')
        
        plotter.set_background('white')
        
        out_file = Path(output_dir) / "campaign_3d.png"
        plotter.screenshot(str(out_file))
        plotter.close()
    except Exception as e:
        print(f"Failed to generate PyVista plot: {e}")
