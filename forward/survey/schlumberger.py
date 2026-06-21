def generate_schlumberger(num_electrodes):
    """
    Generates Schlumberger array sequence.
    A M N B format.
    """
    sequence = []
    for m_spacing in range(1, num_electrodes // 2):
        for n_spacing in range(1, (num_electrodes - 2 * m_spacing) // 2 + 1):
            for i in range(num_electrodes - 2 * n_spacing - 2 * m_spacing):
                A = i
                M = i + n_spacing
                N = M + 2 * m_spacing
                B = N + n_spacing
                sequence.append((A, B, M, N))
    return sequence
