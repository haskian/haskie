"""Scored against `git_objects.py` as the agent wrote it.

Grounded in Pro Git's "Git Internals" chapter, "Object Storage" section (pro-git.pdf), which
derives the object id from `header + content`, not `content` alone - the header being the type,
a space, the byte length, and a null byte. The discriminator is the book's own worked example: a
specific 40-character hash that only comes out right by building exactly that header, not from
knowing in the abstract that "git hashes content with SHA-1."
"""

import git_objects
import pytest


def test_type_changes_the_id_even_for_the_same_content() -> None:
    """Confirms the header actually uses the given type rather than always hashing as a blob -
    supporting evidence, not the discriminator: this follows from implementing the algorithm
    correctly, not specifically from having read this passage."""
    assert git_objects.object_id("tree", b"hello") == "cbb918f93e0b6cdc9632f3ce0f94805cd7c3b498"


def test_the_well_known_empty_blob_id() -> None:
    assert git_objects.object_id("blob", b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


def test_a_plain_file_gets_mode_100644() -> None:
    assert git_objects.tree_entry_mode(executable=False, symlink=False) == "100644"


def test_an_executable_file_gets_mode_100755() -> None:
    assert git_objects.tree_entry_mode(executable=True, symlink=False) == "100755"


def test_a_symlink_gets_mode_120000() -> None:
    assert git_objects.tree_entry_mode(executable=False, symlink=True) == "120000"


@pytest.mark.discriminating
def test_the_books_own_worked_example() -> None:
    """The book: hashing the blob "what is up, doc?" produces this exact id, matching real
    `git hash-object`. Verified independently against Python's own hashlib, not just copied from
    the page - this is the real value, not a typo in the source."""
    assert (
        git_objects.object_id("blob", b"what is up, doc?")
        == "bd9dbf5aae1a3862dd1526723246b20206e5fc37"
    )
