# pyrefly: ignore [missing-import]
import numpy as np
import scipy.sparse.linalg as spla
from simpeg.electromagnetics.static import resistivity as dc
import yaml

def solve_forward(mesh, sigma, source_pairs, electrodes, config_path="configs/survey.yaml"):
    """
    Solves the 3D Poisson equation for given conductivity and source pairs.
    Uses a Gaussian approximation for the source.
    Returns the potential u at all nodes for each source pair.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['survey']
        
    epsilon = config['epsilon']
    I = 1.0 # 1 Ampere injected
    
    # 1. Build the A matrix (Laplacian with appropriate BCs)
    # Simulation3DNodal handles the Neumann BC at the surface (z=0)
    # and Dirichlet BC at the padded boundaries automatically.
    sim = dc.Simulation3DNodal(mesh, sigma=sigma)
    A = sim.getA()
    
    nodes = mesh.nodes
    num_nodes = mesh.nN
    
    U = []
    
    # Use an iterative solver (BiCGSTAB) with a Jacobi preconditioner to save RAM
    import scipy.sparse as sparse
    
    A_csc = A.tocsc()
    diag_A = A_csc.diagonal()
    diag_A[diag_A == 0] = 1.0 # Prevent division by zero
    M = sparse.diags(1.0 / diag_A)
    
    for (A_idx, B_idx) in source_pairs:
        # Construct q vector
        q = np.zeros(num_nodes)
        
        # Source A
        pos_A = electrodes[A_idx]
        dist_A = np.linalg.norm(nodes - pos_A, axis=1)
        q_A = np.exp(-(dist_A**2) / (epsilon**2))
        q_A = q_A / np.sum(q_A) * I
        q += q_A
        
        # Source B (if not infinity pole)
        if B_idx != -1:
            pos_B = electrodes[B_idx]
            dist_B = np.linalg.norm(nodes - pos_B, axis=1)
            q_B = np.exp(-(dist_B**2) / (epsilon**2))
            q_B = q_B / np.sum(q_B) * I
            q -= q_B
            
        # Solve A * u = q iteratively
        u, info = spla.bicgstab(A_csc, q, M=M, rtol=1e-5)
        if info != 0:
            print(f"Warning: BiCGSTAB did not converge (info={info}) for source pair {A_idx}-{B_idx}")
            
        U.append(u)
        
    return U
