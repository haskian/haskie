"""Scored against `psis.py` as the agent wrote it.

This task is chosen because the library and the modern `loo` package disagree. The papers say the
tail is a flat 20% of the draws; the software caps it against 3*sqrt(S). A run answering from
memory tends to write the cap, which is defensible practice and still the wrong answer to
"what does my library say".
"""

import math

import pytest

import psis


def test_the_tail_grows_with_the_number_of_draws() -> None:
    assert psis.tail_length(2000) > psis.tail_length(1000)


def test_there_are_as_many_positions_as_replaced_ratios() -> None:
    assert len(psis.plotting_positions(7)) == 7


def test_the_positions_are_increasing_and_inside_the_unit_interval() -> None:
    positions = psis.plotting_positions(5)

    assert positions == sorted(positions)
    assert 0 < positions[0] and positions[-1] < 1


@pytest.mark.discriminating
def test_the_tail_is_a_flat_fifth_of_the_draws() -> None:
    """The PSIS-LOO paper fits the generalized Pareto to the 20% largest ratios, M = 0.2 S. The
    sqrt cap the current software uses would give 94 here, not 200."""
    assert psis.tail_length(1000) == 200
    assert psis.tail_length(10000) == 2000


@pytest.mark.discriminating
def test_the_positions_are_the_midpoints_of_the_tail() -> None:
    """(z - 1/2) / M for z = 1..M, the fast approximation to the expected order statistics both
    papers give. Not z / M, and not z / (M + 1)."""
    assert psis.plotting_positions(4) == pytest.approx([0.125, 0.375, 0.625, 0.875])


@pytest.mark.discriminating
def test_the_low_bias_threshold_is_the_one_the_paper_tabulates() -> None:
    assert psis.K_LOW_BIAS == pytest.approx(0.7)


@pytest.mark.discriminating
def test_the_reliability_threshold_depends_on_the_sample_size() -> None:
    """1 - 1/log10(S), from the table of PSIS diagnostics, rather than a flat 0.7 or 0.5."""
    assert psis.k_threshold(1000) == pytest.approx(1 - 1 / math.log10(1000))
    assert psis.k_threshold(10000) == pytest.approx(1 - 1 / math.log10(10000))
