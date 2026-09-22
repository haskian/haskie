"""Reference for `psis_smoothing`, from vehtari-2015-practical-bayesian... p.4 and the PSIS
paper's diagnostics table, p.14."""

import math

K_LOW_BIAS = 0.7  # "Maximum khat for low bias in Pareto smoothed estimate", PSIS Table 1


def tail_length(S: int) -> int:
    """M = 0.2 S: the LOO paper fits the generalized Pareto to the 20% largest ratios."""
    return int(0.2 * S)


def plotting_positions(M: int) -> list[float]:
    """(z - 1/2) / M for z = 1..M: Blom's approximation to the expected order statistics."""
    return [(z - 0.5) / M for z in range(1, M + 1)]


def k_threshold(S: int) -> float:
    """"Maximum khat for reliable Pareto smoothed estimate", PSIS Table 1."""
    return 1 - 1 / math.log10(S)
