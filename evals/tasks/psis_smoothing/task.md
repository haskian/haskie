Write `psis.py` in the current directory, implementing the parts of Pareto-smoothed importance
sampling that Vehtari, Gelman and Gabry, "Practical Bayesian model evaluation using leave-one-out
cross-validation and WAIC", and Vehtari et al., "Pareto Smoothed Importance Sampling", pin down
exactly. Standard library only — no numpy, and no fitting code.

Define exactly these names:

    def tail_length(S: int) -> int:
        """How many of the largest importance ratios the generalized Pareto is fitted to, for S
        posterior draws, as the PSIS-LOO paper specifies it."""

    def plotting_positions(M: int) -> list[float]:
        """The quantile positions at which the fitted distribution's inverse CDF is evaluated to
        replace the M largest ratios, in the order the papers give them."""

    K_LOW_BIAS: float
        # the khat above which the smoothed estimate is no longer low-bias

    def k_threshold(S: int) -> float:
        """The sample-size-dependent maximum khat for a reliable smoothed estimate."""

Take each value from those papers rather than from the current `loo` implementation — they do not
agree on all of them — and say where each came from. If you cannot confirm a value against a
source, write your best estimate rather than stopping, and mark that line `# unconfirmed`. Write the file in this
session either way; do not stop to ask.
