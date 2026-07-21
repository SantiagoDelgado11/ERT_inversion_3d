def generate_pole_dipole(num_electrodes_x, num_lines=1, max_n=8):
    """
    Generates Pole-Dipole array sequence for a grid.
    A M N format (B is at infinity).
    We represent the infinity pole B with index -1.
    """
    sequence = []
    for line in range(num_lines):
        offset = line * num_electrodes_x
        for a in range(1, num_electrodes_x // 2):
            for n in range(1, min(max_n + 1, (num_electrodes_x - a) // a + 1)):
                for i in range(num_electrodes_x - (n + 1) * a):
                    A = offset + i
                    B = -1 # Infinity pole
                    M = offset + A - offset + n * a
                    N = offset + M - offset + a
                    sequence.append((A, B, M, N))
    return sequence
