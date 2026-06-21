def generate_dipole_dipole(num_electrodes, max_n=8):
    """
    Generates Dipole-Dipole array sequence.
    A B M N format.
    """
    sequence = []
    for a in range(1, num_electrodes // 3):
        for n in range(1, min(max_n + 1, (num_electrodes - 2*a) // a + 1)):
            for i in range(num_electrodes - (n + 2) * a):
                A = i
                B = A + a
                M = B + n * a
                N = M + a
                sequence.append((A, B, M, N))
    return sequence
