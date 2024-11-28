import random

from locodiff.samplers import get_resampling_sequence

T_values = [10, 20, 30]
r_values = [0, 2, 4, 6]
j_values = [1, 2, 4]


def get_seq_len(T, r, j):
    return j + ((T - j) // j) * (j + 2 * j * (r - 1)) + (T - j) % j


for _ in range(30):
    T = random.choice(T_values)
    r = random.choice(r_values)
    j = random.choice(j_values)
    sequence = get_resampling_sequence(T, r, j)
    net_down = sequence.count("down") - sequence.count("up")
    assert len(sequence) == get_seq_len(
        T, r, j
    ), f"{len(sequence)}, {get_seq_len(T, r, j)}"
    assert (
        net_down == T
    ), f"Invalid sequence\n sequence: {sequence}\n net down: {net_down}\n T: {T}, r: {r}, j: {j}"
