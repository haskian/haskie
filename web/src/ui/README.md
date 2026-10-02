# ui

Shared components for the haskie web app. Each one reproduces a block of the
`design/` pages: same elements, same class names, same ARIA attributes.

`design/design.css` at the repository root is the single source of truth for
tokens, palette, typography and component styles. `src/index.css` imports it and
holds nothing else. Never copy a rule out of `design.css` into this folder, and
never add a stylesheet here. Page-only layout belongs in `src/pages/<Page>.css`.

## Import

```tsx
import { Field, Picker, Tabs, Tile } from '../ui'
```

`vite.config.ts` sets `server.fs.allow: ['..']` so the dev server may read
`design/design.css` and `design/haskie-logo.svg` from above the Vite root.

## Components

| Component | Usage | What it is |
| --- | --- | --- |
| `Shell` | `<Shell current="documents" counts={counts} side={<Groups />}>{page}</Shell>` | Page frame, logo, nav |
| `Logo` | `<Logo />` | The mark and wordmark, linking to Explore |
| `Statusbar` | `<Statusbar status={status} />` | Fixed strip; polls `/api/operations/activity` itself |
| `Picker` | `<Picker options={scopes} value={scope} onChange={setScope} />` | `<details>` dropdown, each option a label plus a `sub` |
| `Tabs` | `<Tabs tabs={[{ id: 'match', label: 'Match' }]} selected={tab} onSelect={setTab} />` | The strip only; the caller renders the panels |
| `Modal` | `<Modal open={open} onClose={close} title={doc} subtitle="collection">{panels}</Modal>` | Native `<dialog>` |
| `Tile` | `<Tile icon={FileText} name={doc.name} sub={sub} description={doc.description} cover={api.coverUrl(doc.name)} onClick={open} />` | One card in a gallery, with an optional cover behind it. On hover, `hint` shows below the name and detail, `description` alone |
| `GallerySection` | `<GallerySection label="A–E" large>{tiles}</GallerySection>` | One lettered band of tiles |
| `SearchBox` | `<SearchBox value={q} onChange={setQ} onSubmit={run} placeholder="Search" scope={<Picker … />} />` | The one search or filter box; a filter passes `onChange` alone |
| `SearchTook` | `<SearchTook counts="12 chunks · 3 sections" ms={took} />` | The line above the results |
| `HitGrid` | `<HitGrid results={hits} query={q} onOpen={open} />` | Chunks, passages or excerpts; one card shape |
| `SectionGrid` | `<SectionGrid sections={found.sections} onOpen={setOpened} />` | The map's sections, one card each with its descriptors and the related sections under it |
| `MapDocuments` | `<MapDocuments documents={found.documents} />` | The documents a map reached, best first, each a link to its page |
| `SectionsModal` | `<SectionsModal section={opened} onClose={close} onOpen={setOpened} />` | One section of the map: what the map said about it, and its document at its heading |
| `SearchPanel` | `<SearchPanel run={(q) => api.explore(q, 'passage', { collections: [name] })} placeholder="Search this collection" plural="passages" />` | Box, results, the search's timings and the match modal with its score lineage, for one scope |
| `MatchModal` | `<MatchModal match={open} query={q} scoring={how} onClose={close} />` | One result, and the document it came from. A chunk shows as the models read it: its frame, its typed pieces, the cut reason on each side and its sizes. A passage shows its text. `scoring` (the `X-Score-Lineage` header) is the hint beside the score |
| `Jobs` | `<Jobs jobs={jobs} variant="glass" stripes />` | One weighted bar per job of an operation |
| `Kv` | `<Kv rows={[['Status', doc.status], ['Size', bytes.format(doc.size)]]} />` | Key and value rows |
| `Field` | `<Field label="Chunk size" help={docs['conversion.chunk_size'].description}><input className="input" /></Field>` | A labelled form control with the setting's help text |
| `Toggle` / `Check` | `<Toggle label="Classic background" checked={on} onChange={setOn} />` | `Toggle` renders `role="switch"` |
| `Mark` | `<Mark text={hit.text} query={q} />` | Wraps the query's terms in `<mark>` |
| `DocumentPanes` | `<DocumentPanes doc={name} preview={doc.preview} />` | Source pane plus streamed markdown |
| `Skeleton` | `<Skeleton />` | The shape of a document while it loads |
| `DescriptionBox` | `<DescriptionBox value={doc.description} placeholder="What is it about?" onSave={save} />` | Saves on blur and on unmount, only what changed |
| `Info` | `<Info>In no collection yet.</Info>` | A line that says how things are, boxed in a gray tint with an info icon. Not for errors |
| `RenameForm` | `<RenameForm key={name} name={name} label="Collection name" busy={busy} onRename={rename} />` | A name input with a Rename button; keyed by the name, so each rename starts a new draft |
| `DropOverlay` | `<DropOverlay onFiles={upload} />` | Document-level drag listeners plus the overlay |
| `ModelFacts` / `EmbedderFacts` | `<ModelFacts name={name} metadata={metadata} />`, `<EmbedderFacts model={model} metadata={metadata} />` | A model's facts under its picker; nothing for the full-text-only profile |
| `Num` / `SearchField` | `<Num … />`, `<SearchField … />` | A number setting in the design's `.field`; one search setting, as Settings and the first run show it |

## Modules

Pure helpers. No JSX, so a page may import one without pulling a component in.
The barrel `../ui` exports the ones pages use. The rest (`piecesOf`, `frameOf`,
`chunkSizes`, `CUT_REASONS`, `PIECE_NAMES`, `HEADING_SEP`, `headingPath`,
`timeline`) are imported from `ui/match`, `ui/anchor` or `ui/timeline` directly,
mostly by `ui/` itself.

| Module | Usage | What it is |
| --- | --- | --- |
| `documents` | `documentIcon(doc.suffix)`, `nameRange(name)`, `groupByRange(rows, (row) => row.name)` and `NAME_RANGES` | The icon and the gallery bands every listing uses |
| `searchFields` | `effectiveSearch(overrides, defaults)`, `visibleSearchFields(effective)`, `visibleExpansionFields(effective)` and `SEARCH_BOUNDS` | The settings a search runs with, which search and expansion fields a form asks for, and the legal numeric bounds |
| `match` | `position(match)`, `headingOf(match)`, the kind guards, the `Match` type; `piecesOf`, `frameOf`, `chunkSizes`, `CUT_REASONS`, `PIECE_NAMES`; `HEADING_SEP` | Where a result sits in its document. A chunk as the chunk view draws it. The heading path separator the backend uses |
| `anchor` | `headingPath(toc, index)` and `anchorIndex(toc, anchor)` | The heading path down to a table-of-contents entry, and the entry a result opens at |
| `markTerms` | `markTerms(text, query)` | The list `Mark` renders |
| `timeline` | `timeline(steps)` | A search's step timings as the server ran them, with the branches of several questions side by side, for `SearchTook` |
| `options` | `choices`, `docFor`, `profileOptions`, `rerankerOption`, `rerankerOptions` | Setting docs and model pickers' options |

Icons come from `lucide-react` and always carry `className="icon"`; the design
sheet sizes them. Pass the component itself where a component takes an icon:
`icon={FileText}`.

## Tests

`bun test` renders the components with `react-dom/server` and asserts on the
markup. No DOM, no testing-library. `DocumentPanes` streams from the API, so it
is left out.
