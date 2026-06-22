import yaml

def generate_dipole_dipole(num_electrodes, max_n=8):
    """
    Generates Dipole-Dipole array sequence.
    A B M N format.
    """
    with open("configs/survey.yaml", 'r') as f:
        config = yaml.safe_load(f)['survey']
        
    dd_config = config['arrays'].get('dipole_dipole', {})
    a_max = dd_config.get('a_max', 6)
    n_max = dd_config.get('max_n', max_n)
    
    sequence = []
    for a in range(1, a_max + 1):
        for n in range(1, n_max + 1):
            for i in range(num_electrodes):
                A = i
                B = i + a
                M = B + n * a
                N = M + a
                
                if N < num_electrodes:
                    sequence.append((A, B, M, N))
    return sequence
