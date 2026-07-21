def generate_wenner(num_electrodes_x, num_lines=1):
    """
    Generates Wenner array sequence for a grid.
    A M N B format. Spacing a=1,2,3...
    Returns a list of tuples (A_idx, B_idx, M_idx, N_idx) 0-indexed.
    """
    sequence = []
    for line in range(num_lines):
        offset = line * num_electrodes_x
        for a in range(1, num_electrodes_x // 3 + 1):
            for i in range(num_electrodes_x - 3 * a):
                A = offset + i
                M = offset + i + a
                N = offset + i + 2 * a
                B = offset + i + 3 * a
                sequence.append((A, B, M, N))
    return sequence
