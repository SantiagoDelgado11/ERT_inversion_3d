def generate_schlumberger(num_electrodes_x, num_lines=1):
    """
    Generates Schlumberger array sequence for a grid.
    A M N B format.
    """
    sequence = []
    for line in range(num_lines):
        offset = line * num_electrodes_x
        for m_spacing in range(1, num_electrodes_x // 2):
            for n_spacing in range(1, (num_electrodes_x - 2 * m_spacing) // 2 + 1):
                for i in range(num_electrodes_x - 2 * n_spacing - 2 * m_spacing):
                    A = offset + i
                    M = offset + i + n_spacing
                    N = offset + M - offset + 2 * m_spacing
                    B = offset + N - offset + n_spacing
                    sequence.append((A, B, M, N))
    return sequence
