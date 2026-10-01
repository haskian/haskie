"""Sections: every document's runs of chunks under one heading path, as a tree, each with the
words that say what it is about.

`build.py` names a document's sections and their ids from its chunks and describes them;
`descriptors.py` holds the strategies that pick each section's descriptors. The embedding cache
keeps them, one file per cached embedding (`embed_cache.sections_path`).
"""
