"""Outline: every document's sections as a tree, each with the words that say what it is about.

`build.py` turns a document's cached chunks into the tree (`Node`), once per document and
embedding model, with a vector per node; `keywords.py` holds the strategies that pick each
node's keywords and the word statistics they and the search's map of sections
(`search/overview.py`) share; `store.py` keeps the tree in a JSON file beside the document's
markdown and its nodes, with their vectors, in one LanceDB index across every collection.
"""
