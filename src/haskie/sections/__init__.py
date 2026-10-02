"""Sections: every document's runs of chunks under one heading path, as a tree, each with the
words that say what it is about.

`build.py` names a document's sections and their ids from its chunks and describes them, by a
strategy: `descriptors.py` holds the protocol and c-TF-IDF, `generated.py` the one a language model
writes. The embedding cache keeps them, one file per cached embedding (`embed_cache.sections_path`).
"""
