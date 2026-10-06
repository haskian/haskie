# Open corpora

Collections of openly available books, one per subject, for benchmarks wider than the seven
bookqa sources. The books come from free-programming-books' subject list, pinned to one commit,
and every book on that list is allowed. No book enters the repository: `manifest.json` pins each
one by URL and SHA-256, and setup downloads it into the gitignored `evals/corpus/open/`.

```sh
mise run eval:corpora:build                                       # pick the books, write the manifest
mise run eval:corpora:build -- --subject "Compiler Design" --per-collection 10
mise run eval:corpora:setup                                       # download, check, index
mise run eval:corpora:setup -- --collection open-operating-systems
```

## Picking the books

`build` reads the subjects asked for (`--subject`, repeatable; the default is the pilot's three)
and tries each subject's links in the list's order until it keeps `--per-collection` books
(default 10). A link is a candidate when it points at one PDF or EPUB, by its extension or by the
`(PDF)` or `(EPUB)` the list writes after it. A book is kept when it:

- downloads, after one retry, so a passing network error does not drop it;
- is under 100 MB;
- is a PDF whose first 15 pages hold at least 2,000 characters of text (not a scan, not an HTML
  page behind a .pdf link), or a real EPUB;
- is not a byte-for-byte copy of a book already kept;
- is not in `build.EXCLUDED`: books that pass these checks but haskie cannot read. pypdf and
  haskie's converter extract text differently (a book pypdf reads as symbols, haskie may read
  fine, and the reverse), so only an import tells. When `setup` fails on a book, add it there.

A link that answers with an HTML page, a book's landing page, is followed one step, to the PDF or
EPUB on it whose file name shares the most words with the book's title. The manifest then pins
that file's URL and names the page in `page`.

Building a subject again replaces its collection in the manifest and leaves the others alone, so
subjects can be added one at a time. Every link tried and left out goes into the manifest's
`skipped`, with the reason: a dead host,
a 404 or 403, a page with no file on it, a file too large.

## Setup

`setup` starts this suite's own haskie instance (`evals/.haskie-eval-corpora`, port 8126,
`HASKIE_EVAL_CORPORA_HOME` and `HASKIE_EVAL_CORPORA_URL`), with the embedding profile the other
evals use (`granite-small-english`). It downloads whatever is missing and refuses a file whose
hash no longer matches the manifest, since questions written from it would no longer hold. Each
manifest collection becomes one haskie collection, named `open-<subject>`. Like every eval setup,
it waits for each collection's post-index maintenance before it returns.

## Layout

| file | what |
| --- | --- |
| `build.py` | the list, the candidates, the checks, landing pages, the manifest |
| `corpus.py` | the manifest's types, downloads checked by hash, indexing |
| `manifest.json` | the books: collections, files, hashes, and the links left out; versioned |
