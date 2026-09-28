# Documents and collections

A document and a collection are separate things. A document is imported once and belongs to no
collection. A collection is a named set of documents with its own LanceDB table and its own chunk
and search settings. One document can sit in any number of collections.

```mermaid
erDiagram
    DOCUMENT ||--o{ MEMBERSHIP : "is in"
    COLLECTION ||--o{ MEMBERSHIP : holds
    DOCUMENT ||--o{ EMBEDDING : "is cached as"
    DOCUMENT {
        string name "fixed at import"
        string status "queued ... imported"
        string parser
        string description
    }
    COLLECTION {
        string name
        string overrides "chunk and search settings"
        string description "what the agent reads to choose"
    }
    MEMBERSHIP {
        string status "pending ... indexed"
    }
    EMBEDDING {
        string id "sha256 of the URN"
        string urn "document, model, chunk settings, parser, OCR policy"
    }
```

A membership has no column pointing at an embedding. When a collection indexes a document, it
computes the cache id from its own chunk settings and reads that entry.

## A document's life

An upload from the UI lands in `staging/` first, as a file and a `staging` row, with no document
yet. The import then fixes the name, creates the document and moves the file into its folder. A
name is taken ignoring case: the folder is named after it, and on a case-insensitive disk
`Notes.md` and `notes.md` would be one folder. An agent's `add_document` imports a local path directly and copies the file. When it refuses the
path, the error names the file alone. The audit trail copies that error, and it never records the
folder an import came from. The nightly run (at 03:17, if haskie is running then) sweeps uploads
older than a day.

```mermaid
stateDiagram-v2
    [*] --> queued: import
    queued --> converting
    converting --> embedding: markdown written
    embedding --> imported: default embedding cached
    queued --> error
    converting --> error
    embedding --> error
    queued --> cancelled
    converting --> cancelled
    embedding --> cancelled
    error --> queued: re-import
    cancelled --> queued: re-import
    queued --> deleting: delete
    converting --> deleting: delete
    embedding --> deleting: delete
    imported --> deleting: delete
    error --> deleting: delete
    cancelled --> deleting: delete
    deleting --> [*]: detached everywhere, folder dropped
```

Any failure lands in `error`: a parser error, an OCR policy failure, or retries run out. A
re-import runs from `queued`, `error` or `cancelled`, with the `parser` and `skip_ocr_pages` the
document was imported with: to change either, delete it and import it again. A delete is accepted
in any state. The original suffix is kept in the name, because it decides the route:

- PDFs convert page by page with pdf-inspector.
- Text and HTML files are read as they are.
- Images are stored and previewed, with no text to index.
- Everything else converts with anydoc, or is read as raw text when the document's `parser` is
  `plain`.

The import also warms the embedding cache for the default chunk settings. A re-import clears the
document's cached embeddings first.

## Repeats

The web UI adds one document at a time, so each new book gets checked for repeats:

- **The same file.** Staging and a path import both take the MD5 of the bytes. Staging answers
  with `duplicates`, the documents that already hold those bytes. The UI then suggests discarding
  the upload, and the import button turns into "Import anyway".
- **The nearest documents.** Writing a cache entry also stores the document as one vector: the
  mean of its unit chunk vectors, normalized. `GET /api/documents/{name}/similar` names the
  identical documents and the three nearest by cosine, under the current embedding model. The
  vector exists only once the import has embedded the document, so the UI follows the new book
  until then. Full-text only has no vectors, so it finds no nearest documents.

## A membership's life

Attaching a document to a collection is an index operation with its own status, per collection.

```mermaid
stateDiagram-v2
    [*] --> pending: attach
    pending --> indexing
    indexing --> indexed: rows written to the collection's table
    indexing --> error
    pending --> cancelled
    indexing --> cancelled
    indexed --> indexing: re-index or Index all
    error --> indexing: re-index or Index all
    cancelled --> indexing: re-index or Index all
    pending --> removing: detach
    indexing --> removing: detach
    indexed --> removing: detach
    error --> removing: detach
    cancelled --> removing: detach
    removing --> [*]: rows and membership removed
    removing --> error: the removal failed
```

Detaching deletes the document's rows from that collection's table. The request marks the
membership `removing`, cancels its index and queues the removal on the collection's single writer.
It answers at once: a compaction or another document's write may hold that writer for minutes.
`removing` counts as active, so the UI keeps polling until the membership is gone. Until then, the
old rows stay in the table, but a search leaves them out. The same holds for a document being
deleted, in every collection. An attach or a re-index of that document is refused
meanwhile, and its index can no longer change the status. A removal that fails leaves the
membership in `error` with the reason, and detaching again retries it. Deleting a collection deletes
its table and memberships, and keeps every document. Deleting a document detaches it from every
collection first, then drops its folder and row. Memberships also go when their collection or
document is deleted.

Renaming a collection moves its row, its memberships, every session that chose it and its folder
in one transaction. The index table holds no collection name, so it moves as it is. A rename is
refused while any work of the collection runs: an index write or a maintenance run still holds
the old name, and would put the old folder back.

## The embedding cache

The cache makes one document cheap to share between collections. Each computed embedding is one parquet file
under the document, plus one `embeddings` row. Its id is the SHA-256 of a URN, one line, that
names everything the rows depend on:

```
document:<name>;model:<model>;chunk_size:<n>;chunk_merge_below:<n>;chunk_frame:<b>;chunker:<c>;chunk_version:<v>;parser:<p>;skip_ocr_pages:<b>
```

`<model>` is `EmbeddingModel.cache_name`: the model's name and vector size, plus a hash of its
document prefix and Matryoshka recipe. Those are everything that shapes a stored vector, so a
change to any of them misses the cache and needs no `chunk_version` bump.

Same inputs give the same id, so the work runs once, until a re-import clears it. Two collections
that ask for the same missing entry at the same moment share one DBOS run. A collection with other
chunk settings gets its own entry. The accelerator, the query prefix and the duplicate
thresholds are not in the key: they shape no stored vector.

```mermaid
flowchart LR
    attach["attach document<br/>to collection"] --> key["URN from the collection's<br/>chunk settings + model"]
    key --> run["ensure_embedding<br/>(one run per id)"]
    run --> hit{"cached?"}
    hit -- yes --> write["write rows into<br/>the collection's table"]
    hit -- no --> embed["chunk + embed"] --> write
```

Code: `document/document.py`, `collection/collection.py`, `indexing/embed_cache.py`.
