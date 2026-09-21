# ui

Shared components for the haskie web app. Each one reproduces a block of the
`design/` pages: same elements, same class names, same ARIA attributes.

`design/design.css` at the repository root is the single source of truth for
tokens, palette, typography and component styles. `src/index.css` imports it and
holds nothing else. Never copy a rule out of `design.css` into this folder, and
never add a stylesheet here: page-only layout belongs in `src/pages/<Page>.css`.

## Import

```tsx
import { Field, Picker, Tabs, Tile } from '../ui'
```

`vite.config.ts` sets `server.fs.allow: ['..']` so the dev server may read
`design/design.css` and `design/haskie-logo.svg` from above the Vite root.

## Components

| Component | Usage |
| --- | --- |
| `Shell` | `<Shell current="documents" counts={counts} side={<Groups />}>{page}</Shell>` — page frame, logo, nav. |
| `Logo` | `<Logo />` — the mark and wordmark, linking to Explore. |
| `Statusbar` | `<Statusbar status={status} />` — fixed strip; polls `/api/jobs/activity` itself. |
| `Picker` | `<Picker options={scopes} value={scope} onChange={setScope} />` — `<details>` dropdown, each option a label plus a `sub`. |
| `Tabs` | `<Tabs tabs={[{ id: 'match', label: 'Match' }]} selected={tab} onSelect={setTab} />` — the strip only; the caller renders the panels. |
| `Modal` | `<Modal open={open} onClose={close} title={doc} subtitle="collection">{panels}</Modal>` — native `<dialog>`. |
| `Tile` | `<Tile icon={FileText} name={doc.name} sub={doc.description} hint={doc.description} onClick={open} />`. |
| `GallerySection` | `<GallerySection label="A–E" large>{tiles}</GallerySection>`. |
| `HitGrid` | `<HitGrid hits={hits} query={q} onOpen={open} />` or `<HitGrid matches={matches} query={q} />`. |
| `SearchPanel` | `<SearchPanel run={(q) => api.searchCollection(name, q)} placeholder="Search this collection" />` — box, hits and match modal for one scope. |
| `MatchModal` | `<MatchModal hit={open} query={q} onClose={close} />` — one result: the passage, and the document it came from. |
| `Stages` | `<Stages stages={stages} variant="glass" stripes />` — one weighted bar per stage of a job. |
| `Kv` | `<Kv rows={[['Status', doc.status], ['Size', bytes.format(doc.size)]]} />`. |
| `Field` | `<Field label="Chunk size" help={docs['conversion.chunk_size'].description}><input className="input" /></Field>`. |
| `Toggle` / `Check` | `<Toggle label="Classic background" checked={on} onChange={setOn} />` — `Toggle` renders `role="switch"`. |
| `Mark` | `<Mark text={hit.text} query={q} />` wraps the query's terms in `<mark>`. |
| `DocumentPanes` | `<DocumentPanes doc={name} preview={doc.preview} />` — source pane plus streamed markdown. |
| `DropOverlay` | `<DropOverlay onFiles={upload} />` — window drag listeners plus the overlay. |

## Modules

Pure helpers the pages share. No JSX, so a page may import one without pulling a
component in.

| Module | Usage |
| --- | --- |
| `documents` | `documentIcon(doc.suffix)`, `nameRange(name)`, `groupByRange(rows, (row) => row.name)` and `NAME_RANGES` — the icon and the gallery bands every listing uses. |
| `searchFields` | `effectiveSearch(overrides, defaults)` and `visibleSearchFields(effective)` — what a search actually runs with, and which fields a form asks for. |
| `match` | `position(hit)` and the `Match` type — where a result sits in its document. |
| `markTerms` | `markTerms(text, query)` — the list `Mark` renders. |

Icons come from `lucide-react` and always carry `className="icon"`; the design
sheet sizes them. Pass the component itself where a component takes an icon:
`icon={FileText}`.

## Tests

`bun test` renders the pure components with `react-dom/server` and asserts on
the markup. No DOM, no testing-library. `Statusbar` and `DocumentPanes` fetch,
so they are left out.
