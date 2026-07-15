def generate_dipole_dipole(num_electrodes, max_n=8, a_max=6):
    """
    Generates Dipole-Dipole array sequence.
    A B M N format.
    """
    sequence = []
    for a in range(1, a_max + 1):
        for n in range(1, max_n + 1):
            for i in range(num_electrodes):
                A = i
                B = i + a
                M = B + n * a
                N = M + a
                
                if N < num_electrodes:
                    sequence.append((A, B, M, N))
    return sequence
