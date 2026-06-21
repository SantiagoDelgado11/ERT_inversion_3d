import numpy as np

def calculate_geometric_factor(pos_A, pos_B, pos_M, pos_N):
    """
    Calculates the half-space geometric factor K for a 4-electrode setup.
    B_idx = -1 means B is at infinity.
    pos_B can be None if B is at infinity.
    """
    r_AM = np.linalg.norm(pos_A - pos_M)
    r_AN = np.linalg.norm(pos_A - pos_N)
    
    term_AM = 1.0 / r_AM if r_AM > 0 else 0
    term_AN = 1.0 / r_AN if r_AN > 0 else 0
    
    if pos_B is not None:
        r_BM = np.linalg.norm(pos_B - pos_M)
        r_BN = np.linalg.norm(pos_B - pos_N)
        term_BM = 1.0 / r_BM if r_BM > 0 else 0
        term_BN = 1.0 / r_BN if r_BN > 0 else 0
    else:
        term_BM = 0
        term_BN = 0
        
    geom_factor = 2 * np.pi / (term_AM - term_BM - term_AN + term_BN)
    return geom_factor

def extract_measurements(mesh, U, source_pairs, measurement_sequences, electrodes):
    """
    Interpolates node potentials to electrode positions and calculates Delta V
    and apparent resistivity.
    U is a list of potentials corresponding to each source_pair.
    measurement_sequences is a list of lists of (M, N) tuples per source pair.
    """
    # Create interpolation matrix for all electrodes
    P = mesh.get_interpolation_matrix(electrodes, 'N')
    
    results = []
    
    for i, (A_idx, B_idx) in enumerate(source_pairs):
        u = U[i]
        v_electrodes = P @ u
        
        for (M_idx, N_idx) in measurement_sequences[i]:
            v_M = v_electrodes[M_idx]
            v_N = v_electrodes[N_idx]
            delta_v = v_M - v_N
            
            pos_A = electrodes[A_idx]
            pos_B = electrodes[B_idx] if B_idx != -1 else None
            pos_M = electrodes[M_idx]
            pos_N = electrodes[N_idx]
            
            K = calculate_geometric_factor(pos_A, pos_B, pos_M, pos_N)
            rho_a = delta_v * K # I is 1.0
            
            results.append({
                'A': A_idx, 'B': B_idx, 'M': M_idx, 'N': N_idx,
                'delta_v': delta_v,
                'rho_a': rho_a,
                'K': K
            })
            
    return results
