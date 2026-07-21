def generate_dipole_dipole(num_electrodes_x, num_lines=1, max_n=8, a_max=6):
    """
    Generates Dipole-Dipole array sequence for a grid.
    A B M N format.
    """
    sequence = []
    for line in range(num_lines):
        offset = line * num_electrodes_x
        for a in range(1, a_max + 1):
            for n in range(1, max_n + 1):
                for i in range(num_electrodes_x):
                    A = offset + i
                    B = offset + i + a
                    M = offset + B - offset + n * a
                    N = offset + M - offset + a
                    
                    if N - offset < num_electrodes_x:
                        sequence.append((A, B, M, N))
    return sequence
