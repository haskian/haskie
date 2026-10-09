# Curate and explore your library

Use the web UI to choose the sources agents can consult. Mix technical references, domain
documents and examples of the approaches you prefer. [Setup](setup.md) covers installation,
models and agent connections.

## Import documents once

Drop files onto **Documents**. Review their names before import, then import the batch.
haskie converts documents to Markdown and prepares them for search in the background. Text and
HTML formats are read as supplied. Use the side-by-side preview to check the conversion.

The import view warns when the same file is already in the library. Reuse that document rather
than import another copy. Once embedding finishes, the similar-document view can suggest nearby
documents. Similarity suggests where to look; it does not establish that their contents agree.

A document's name is fixed at import. A description can explain its scope and help an agent
choose it. You can write the description yourself or request **Describe with AI**. Automatic
descriptions during indexing preserve descriptions you wrote; an explicit generation request
replaces the current description. See [documents and collections](documents-and-collections.md)
for import states, duplicate handling and shared caches.

## Organize collections around the work

Create a collection for a topic, product, project or preferred approach. Add documents already
in your library. One document can belong to several collections without another import.
Removing it from a collection keeps the document; deleting the document removes it everywhere.

Describe the collection so an agent knows what it covers. **Describe with AI** can summarize its
members' descriptions. Review the result against the scope you intended.

Collections inherit your chunking and search settings unless you override them. They share an
embedding run when the document, model and chunk settings match. Different chunk settings need
their own cached representation. After changing the embedding profile, use **Index all** in each
collection. See [setup](setup.md#choose-models-and-search-settings).

## Explore what the agent receives

Ask a question on **Explore** and choose the view that suits it:

| View | Use it to |
| --- | --- |
| Sections | Map a topic across documents. Read headings and descriptors before choosing sections to open. |
| Excerpts | Read focused evidence with source locations, as an agent would. |
| Passages | Inspect how neighbouring matched chunks form a readable passage. |
| Chunks | Inspect the indexed units and their search scores. |

Open a section to inspect the map's coverage and follow it to the document's heading. A useful
section can appear under `related` even when it was not selected as a main result.

`related` and `also_in` serve different purposes. A related section is a candidate for more
reading. An `also_in` entry passed a repeat test and preserves the location of repeated evidence.
It may be elsewhere in the same document. Check the document name before counting it as another
source. See [folding repeats](search.md#folding-repeats).

Citations carry a document, heading and location. For example, `manual.pdf p.12 L240-265` means
page 12 and lines 240–265 in the full converted Markdown file. Page references apply to PDFs;
line numbers count through the whole Markdown document. A citation lets you inspect the evidence
in context. A high search score does not prove that it answers your question.

## Review activity and improve coverage

| Page | What it shows |
| --- | --- |
| Operations | Background imports, indexing, descriptions and maintenance, with progress, errors and cancellation. |
| Sessions | Agent search and tool activity associated with each session. |
| Gaps | Questions the library did not answer, grouped by topic. |
| Insights | Search activity and indexed chunks over time. |
| Settings | Defaults, model choices and explanations of individual controls. |

After adding sources for a gap, replay its questions. Review the new evidence before resolving
the gap. Dismiss questions outside the library's intended scope. A gap is a signal to inspect,
not proof that no relevant passage exists. The [Gaps guide](gaps.md) explains detection and replay.

Indexing uses a CPU budget and resumes interrupted work. Large PDFs run in batches, and matching
collections reuse cached embeddings. The [indexing guide](indexing.md) explains recovery and
resource limits. Model status distinguishes downloading, loading, ready and failed models; see
[runtime](runtime.md#model-loads).

## Supported formats

| Kind | Extensions |
| --- | --- |
| PDF | `.pdf`; extracted page numbers are retained for citations. |
| Office | `.doc` `.docx` `.docm` `.ppt` `.pptx` `.pptm` `.pps` `.ppsx` `.ppsm` `.pot` `.xls` `.xlsx` `.xlsm` `.xlsb` |
| OpenDocument | `.odt` `.ods` `.odp` |
| Other documents | `.epub` `.rtf` |
| Text | `.md` `.markdown` `.txt` `.csv` `.json` `.html` `.htm` |
| Images | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.svg`; their text is read with OCR, SVG aside. |

Optical character recognition (OCR) runs on your device, with PP-OCRv6 small. It reads PDF
pages that carry no text, such as scans, and images. Its model, about 31 MB, downloads with the
other models while the Read scans with OCR setting is on (the default). A PDF page OCR reads no
text on, such as a blank page, is skipped by default and the rest is indexed. A PDF with no text
on any page fails with an explanation. Disabling the skip setting makes any such page fail the
import. An image OCR reads no text on is imported with nothing to search.
A PDF's info shows how many of its pages converted, how many of those OCR read, and how many
were not converted.

Documents imported before haskie had OCR keep their old markdown: their images and skipped
scans stay unsearchable. Delete such a document and import it again to read them.

## Where to go next

- [MCP and agent integrations](mcp.md): tool capabilities, scope, sessions and citations.
- [Setup and operation](setup.md): commands, model choices, hardware and upgrades.
- [Why haskie exists](why-haskie.md): the product choices and their research sources.
