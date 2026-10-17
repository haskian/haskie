"""Registers the marker every task's `test_task.py` uses, so pytest doesn't warn about it and
`-m discriminating` filtering is unambiguous."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "discriminating: the assertion that separates having read the source from guessing well",
    )
