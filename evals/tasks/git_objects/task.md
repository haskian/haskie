Write `git_objects.py` in the current directory. Standard library only. Do not run Git or
inspect a repository - compute everything from the arguments given.

Implement two functions from Pro Git's "Git Internals" chapter, "Object Storage" section, which
walks through exactly how `git hash-object` computes an object's key:

    def object_id(object_type: str, content: bytes) -> str:
        """The SHA-1 hex digest git would report for this object - the same value
        `git hash-object` prints, and what `git cat-file -t <id>` would look up."""

    def tree_entry_mode(*, executable: bool, symlink: bool) -> str:
        """The octal mode string git records for a blob's tree entry - a plain file, an
        executable file, or a symbolic link. Exactly one of `executable`/`symlink` is ever
        true; both false means a plain file."""

The book builds the object's key by constructing a header - the object type, a space, the
content's length in bytes, and a trailing null byte - then hashing the header concatenated with
the content, not the content alone.

Its own worked example: hashing the blob `b"what is up, doc?"` (17 bytes) this way produces
`bd9dbf5aae1a3862dd1526723246b20206e5fc37`, the same value `echo -n "what is up, doc?" | git
hash-object --stdin` prints.

For tree entries, the book gives the three modes valid for a blob: a plain file is `100644`; an
executable file is `100755`; a symbolic link is `120000` - "the mode is taken from normal UNIX
modes but is much less flexible - these three modes are the only ones that are valid for files
(blobs) in Git."
