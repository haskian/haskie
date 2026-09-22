"""Scored against `editdistance.py` as the agent wrote it.

An in-model task with no corpus overlap at all: the control for the trigger metric. A run that
searches the library here has searched for something no collection could have answered.
"""

import editdistance


def test_the_textbook_pair() -> None:
    assert editdistance.edit_distance("kitten", "sitting") == 3


def test_an_empty_string_costs_the_other_ones_length() -> None:
    assert editdistance.edit_distance("", "abc") == 3
    assert editdistance.edit_distance("abc", "") == 3


def test_identical_strings_cost_nothing() -> None:
    assert editdistance.edit_distance("abc", "abc") == 0


def test_a_substitution_is_one_not_two() -> None:
    assert editdistance.edit_distance("abc", "abd") == 1


def test_the_distance_is_symmetric() -> None:
    assert editdistance.edit_distance("flaw", "lawn") == editdistance.edit_distance("lawn", "flaw")
