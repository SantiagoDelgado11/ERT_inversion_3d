def generate_pole_dipole(num_electrodes, max_n=8):
    """
    Generates Pole-Dipole array sequence.
    A M N format (B is at infinity).
    We represent the infinity pole B with index -1.
    """
    sequence = []
    for a in range(1, num_electrodes // 2):
        for n in range(1, min(max_n + 1, (num_electrodes - a) // a + 1)):
            for i in range(num_electrodes - (n + 1) * a):
                A = i
                B = -1 # Infinity pole
                M = A + n * a
                N = M + a
                sequence.append((A, B, M, N))
    return sequence
