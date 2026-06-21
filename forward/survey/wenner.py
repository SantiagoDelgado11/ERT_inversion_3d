def generate_wenner(num_electrodes):
    """
    Generates Wenner array sequence.
    A M N B format. Spacing a=1,2,3...
    Returns a list of tuples (A_idx, B_idx, M_idx, N_idx) 0-indexed.
    """
    sequence = []
    for a in range(1, num_electrodes // 3 + 1):
        for i in range(num_electrodes - 3 * a):
            A = i
            M = i + a
            N = i + 2 * a
            B = i + 3 * a
            sequence.append((A, B, M, N))
    return sequence
