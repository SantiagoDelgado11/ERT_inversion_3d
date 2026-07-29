import os
import argparse
from pathlib import Path

from dataset_generator.config.settings import load_config
from dataset_generator.geometry.terrain import GeometryManager
from dataset_generator.mesh.builder import MeshGenerator
from dataset_generator.survey.acquisition import SurveyGenerator
from dataset_generator.forward_solver.simulator import ForwardSolver
from dataset_generator.dataset.validator import DatasetValidator
from dataset_generator.dataset.writer import HDF5Writer
from dataset_generator.visualization.plots import plot_sample_summary, plot_3d_model_pyvista

def main():
    parser = argparse.ArgumentParser(description="Generate 3D ERT Dataset")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config. Output directory: {cfg.output_dir}")
    print("Generating a single 3D ERT campaign.")
    
    # Initialize components
    geometry = GeometryManager(cfg.domain)
    survey_gen = SurveyGenerator(cfg.survey, cfg.domain)
    mesh_gen = MeshGenerator(cfg.mesh, cfg.domain)
    solver = ForwardSolver(cfg)
    validator = DatasetValidator(cfg)
    writer = HDF5Writer(cfg)
    
    # Ground truth grid dimensions (resolution of 1m for PINN)
    gt_grid = geometry.get_ground_truth_grid(dx=1.0, dy=1.0, dz=1.0)
    
    print("\n--- Generating Campaign ---")
    try:
        # 1. Generate geometry
        print("1. Generating deterministic anomaly geometry...")
        anomaly = geometry.generate_sphere(cfg.anomaly)
        print(f"   Sphere generated at ({anomaly['x']:.1f}, {anomaly['y']:.1f}, {anomaly['z']:.1f}) with R={anomaly['r']:.1f}m")
        
        # 2. Generate survey combinations
        print("2. Generating survey configurations...")
        survey_data = survey_gen.generate_survey()
        print(f"   Survey generated with {len(survey_data['a_idx'])} measurements.")
        survey_gen.generate_report(survey_data, cfg.output_dir)
        
        # 3. Build Mesh
        print("3. Building and refining TreeMesh...")
        mesh = mesh_gen.build_tree_mesh(survey_data['electrodes'], anomaly)
        active_cells = mesh_gen.get_active_cells(mesh)
        print(f"   Mesh built with {mesh.n_cells} cells ({np.sum(active_cells)} active).")
        
        # 4. Build Model
        print("4. Building physical model...")
        active_model = solver.build_physical_model(mesh, anomaly, active_cells)
        
        # 5. Run Forward Solver
        print("5. Running SimPEG simulation (this may take a while)...")
        survey_data = solver.run_simulation(mesh, active_cells, active_model, survey_data)
        
        # 6. Validate
        print("6. Validating generated data...")
        validator.validate_campaign(survey_data, anomaly, active_model)
        print("   Validation passed.")
        
        # 7. Save Dataset
        print("7. Writing to HDF5 file...")
        writer.save_campaign(survey_data, anomaly, active_model, mesh, gt_grid)
        
        # 8. Visualization
        print("8. Generating visualizations...")
        plot_sample_summary(cfg.output_dir, survey_data, anomaly, cfg.domain)
        plot_3d_model_pyvista(cfg.output_dir, mesh, active_model, anomaly, cfg.domain)
        
        print("Campaign successfully generated and saved.")
        
    except Exception as e:
        print(f"Error generating campaign: {e}")
        # Optionally retry or raise
        raise e

if __name__ == "__main__":
    import numpy as np # need it here for the log messages above
    main()
