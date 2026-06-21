import numpy as np
import yaml
from mesh.mesh_generator import generate_mesh
from geology.conductivity_models import build_conductivity_model
from survey.electrodes import generate_surface_electrodes
from survey.wenner import generate_wenner
from survey.schlumberger import generate_schlumberger
from survey.dipole_dipole import generate_dipole_dipole
from survey.pole_dipole import generate_pole_dipole
from simulation.forward_solver import solve_forward
from simulation.measurements import extract_measurements

def group_by_source(sequence):
    """Groups measurement sequence by source pair (A, B)."""
    grouped = {}
    for (A, B, M, N) in sequence:
        if (A, B) not in grouped:
            grouped[(A, B)] = []
        grouped[(A, B)].append((M, N))
    
    source_pairs = list(grouped.keys())
    measurement_sequences = [grouped[sp] for sp in source_pairs]
    return source_pairs, measurement_sequences

def generate_single_sample(seed=None, mesh=None, config_geology=None, config_survey=None, return_mesh=False):
    """
    Generates a single synthetic sample.
    """
    if seed is not None:
        np.random.seed(seed)
        
    if config_survey is None:
        with open("configs/survey.yaml", 'r') as f:
            config_survey = yaml.safe_load(f)['survey']
            
    if mesh is None:
        mesh = generate_mesh()
        
    sigma, anomalies = build_conductivity_model(mesh, config_geology)
    
    electrodes = generate_surface_electrodes(
        num_electrodes=config_survey['num_electrodes'],
        spacing=config_survey['electrode_spacing'],
        center_x=0.0
    )
    
    # Build complete sequence
    sequence = []
    if config_survey['arrays'].get('wenner', {}).get('active', False):
        sequence.extend(generate_wenner(len(electrodes)))
    if config_survey['arrays'].get('schlumberger', {}).get('active', False):
        sequence.extend(generate_schlumberger(len(electrodes)))
    if config_survey['arrays'].get('dipole_dipole', {}).get('active', False):
        max_n = config_survey['arrays']['dipole_dipole'].get('max_n', 8)
        sequence.extend(generate_dipole_dipole(len(electrodes), max_n))
    if config_survey['arrays'].get('pole_dipole', {}).get('active', False):
        max_n = config_survey['arrays']['pole_dipole'].get('max_n', 8)
        sequence.extend(generate_pole_dipole(len(electrodes), max_n))
        
    source_pairs, measurement_sequences = group_by_source(sequence)
    
    # Solve forward
    U = solve_forward(mesh, sigma, source_pairs, electrodes)
    
    # Extract measurements
    measurements = extract_measurements(mesh, U, source_pairs, measurement_sequences, electrodes)
    
    res = {
        'sigma': sigma,
        'anomalies': anomalies,
        'measurements': measurements,
        'electrodes': electrodes,
        'sequence': sequence
    }
    
    if return_mesh:
        res['mesh'] = mesh
        
    return res
