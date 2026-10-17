"""`run_pytest` reads pytest's own summary line rather than parsing per-test output, so its
whole job is getting that one line right. Each test below runs pytest for real, against a tiny
throwaway test file, rather than a hand-typed string standing in for pytest's output - the bug
this module already shipped once was exactly a wrong guess at that format.
"""

from pathlib import Path

from evals.run import run_pytest


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_all_passing(tmp_path: Path) -> None:
    body = "def test_a(): assert True\ndef test_b(): assert True\n"
    test_file = _write(tmp_path / "test_x.py", body)

    passed, total = run_pytest(test_file, tmp_path)

    assert (passed, total) == (2, 2)


def test_some_failing(tmp_path: Path) -> None:
    """The regression case: pytest's summary reads "N failed, M passed" - failed first - when
    both occur, not "passed" first. A parser that assumed "passed" always comes first, and that
    "failed" is optional and follows it, silently dropped every failure it never looked for."""
    test_file = _write(
        tmp_path / "test_x.py",
        "def test_a(): assert False\n"
        "def test_b(): assert False\n"
        "def test_c(): assert False\n"
        "def test_d(): assert False\n"
        "def test_e(): assert True\n",
    )

    passed, total = run_pytest(test_file, tmp_path)

    assert (passed, total) == (1, 5)


def test_a_collection_error_counts_against_the_total(tmp_path: Path) -> None:
    """A fixture that raises turns every dependent test into a pytest "error", a different
    category from "failed" - and one this eval's grading must not read as tests that never
    existed."""
    test_file = _write(
        tmp_path / "test_x.py",
        "import pytest\n"
        "@pytest.fixture\n"
        "def broken():\n"
        "    raise RuntimeError('setup failed')\n"
        "def test_a(broken): pass\n"
        "def test_b(): assert True\n",
    )

    passed, total = run_pytest(test_file, tmp_path)

    assert (passed, total) == (1, 2)


def test_a_module_that_does_not_import_scores_zero_of_one(tmp_path: Path) -> None:
    """The agent's file was never written, or has a syntax error: pytest reports this as "1
    error", one module that failed to collect, whatever number of tests it would have held. This
    scores 0/1 rather than 0/0 - a collection failure is one real, counted attempt that produced
    no passes, not the absence of an attempt."""
    test_file = _write(tmp_path / "test_x.py", "import a_module_that_does_not_exist\n")

    passed, total = run_pytest(test_file, tmp_path)

    assert (passed, total) == (0, 1)


def test_a_marker_with_no_matches_scores_zero_of_zero_not_a_pass(tmp_path: Path) -> None:
    test_file = _write(tmp_path / "test_x.py", "def test_a(): assert True\n")

    passed, total = run_pytest(test_file, tmp_path, marker="discriminating")

    assert (passed, total) == (0, 0)


def test_a_marker_selects_only_the_matching_tests(tmp_path: Path) -> None:
    test_file = _write(
        tmp_path / "test_x.py",
        "import pytest\n"
        "def test_a(): assert True\n"
        "@pytest.mark.discriminating\n"
        "def test_b(): assert False\n",
    )

    passed, total = run_pytest(test_file, tmp_path, marker="discriminating")

    assert (passed, total) == (0, 1)
