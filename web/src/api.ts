// The wire shapes of the backend (`src/haskie`) and the one client that reads them. Every struct
// is an alias into `schema.d.ts`, which `mise run schema` generates from the Litestar OpenAPI
// document and `mise run check` proves is still current. Nothing here explains what a field
// means, because the definitions the UI shows come from /api/options.docs.
import type { components, operations } from './schema'

// One OpenAPI document describes both directions, so a msgspec field with a default counts as
// "not required" there. On the wire every field is present, and the UI sends whole objects back,
// so the client reads each struct complete, at every depth.
type Complete<T> = T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T
type Wire<K extends keyof components['schemas']> = Complete<components['schemas'][K]>

export type ChunkSettings = Wire<'ChunkSettings'>
export type ConversionSettings = Wire<'ConversionSettings'>
export type CollectionSettings = Wire<'CollectionSettings'>
export type SearchSettings = Wire<'SearchSettings'>
export type SearchOverrides = Wire<'SearchOverrides'>
export type PipelineSettings = Wire<'PipelineSettings'>
export type RetentionSettings = Wire<'RetentionSettings'>
export type UserSettings = Wire<'UserSettings'>
export type FieldDoc = Wire<'FieldDoc'>
// Every choice the UI offers, including the status and kind vocabularies, so no enumeration is
// spelled a second time here. One fetch per page load answers it (see `api.options`).
export type Options = Wire<'Options'>
export type ImportedDocument = Wire<'Document'>
export type Document = Wire<'Listed'>
export type CollectionInfo = Wire<'CollectionInfo'>
export type EmbeddingEntry = Wire<'Entry'> // `embed_cache.Entry`: one cached embedding of a document
export type Hit = Wire<'Hit'>
export type DocumentMatch = Wire<'DocumentMatch'>
export type Status = Wire<'Status'>
export type ModelStatus = Wire<'ModelStatus'>
export type Task = Wire<'Task'>
export type Activity = Wire<'Activity'>
export type Operation = Wire<'Operation'>
export type Job = Wire<'Job'>
export type SessionSummary = Wire<'SessionSummary'>
// Not `Wire`: `EventDetail` is the one struct whose fields really are absent on the wire
// (`omit_defaults`), so completing them would promise fields no action fills.
export type EventDetail = components['schemas']['EventDetail']
export type SessionEvent = Omit<Wire<'SessionEvent'>, 'detail'> & { detail: EventDetail }
export type SearchAt = Wire<'SearchAt'>
export type ChunksAt = Wire<'ChunksAt'>
export type OperationKindSummary = Wire<'OperationKindSummary'>
// Work the backend only accepts (202) and runs in the background. `operation_id` is what a poll of
// `operationProgress` follows.
export type BulkStarted = Wire<'BulkStarted'>
export type OperationProgress = Wire<'OperationProgress'>
export type Preview = Wire<'Preview'>
export type Staged = Wire<'Staged'>
export type Member = Wire<'Member'>
export type CollectionSummary = Wire<'CollectionSummary'>

// The vocabularies the UI narrows on, each read off the field that carries it, so no member is
// spelled out. `Options` lists the same values at runtime, for the dropdowns.
export type Parser = ConversionSettings['parser']
export type Chunker = ChunkSettings['chunker']
export type Accelerator = PipelineSettings['accelerator']
export type EmbeddingProfile = UserSettings['embedding']
export type DocStatus = Document['status']
export type MemberStatus = Member['status']
export type SearchMode = NonNullable<SearchSettings['mode']>
export type Fusion = NonNullable<SearchSettings['fusion']>
export type Reranker = NonNullable<SearchSettings['reranker']>
export type OperationKind = Operation['kind']
export type Stage = Job['stage']
export type RunStatus = Operation['status']
// The three whole-thing operations of the collection kind; `Operation.detail.bulk` carries it.
export type BulkKind = OperationProgress['kind']
// What a session can be seen doing; `collections` is the selection itself being set.
export type SessionAction = SessionEvent['action']

// In lifecycle order, which is the order the Documents page bands them in.
export const DOCUMENT_STATUSES: readonly DocStatus[] = ['queued', 'converting', 'embedding', 'imported', 'error', 'cancelled', 'deleting']
// Still on its way: what the UI polls for and shows a spinner against.
export const ACTIVE_DOCUMENT_STATUSES: readonly DocStatus[] = ['queued', 'converting', 'embedding']
export const ACTIVE_STATUSES: ReadonlySet<RunStatus> = new Set<RunStatus>(['ENQUEUED', 'PENDING'])

// What a collection may override of the chunking defaults; null is "use the default".
export type ChunkOverrides = Pick<CollectionSettings, keyof ChunkSettings>

// The viewer is streamed as NDJSON, so these frames are not a response body and the OpenAPI
// document does not carry them; they mirror `render.Head`, `render.Heading` and `render.Page` by
// hand. One `head`, then one `page` per page of rendered HTML. The HTML comes from the server
// with raw HTML already removed, which is why the pane can insert it; see `haskie/render.py`.
export interface Heading {
  level: number
  text: string
  offset: number
}
export interface Head {
  kind: 'head'
  toc: Heading[]
  preview: Preview | null
  pages: number
}
interface PageFrame {
  kind: 'page'
  number: number | null
  html: string
}
type Frame = Head | PageFrame

export const DEFAULT_PAGE_SIZE = 100
export const MAX_PAGE_SIZE = 1000 // the backend's cap; a bigger page_size is rejected
// The paging parameters `paging.page_request` declares once and every listing takes. OpenAPI has
// no name for a shared parameter set, so they are read off one listing that takes them.
export type PageRequest = Omit<NonNullable<operations['ApiDocumentsListDocuments']['parameters']['query']>, 'status'>
export type Order = NonNullable<PageRequest['order']>
// Litestar names a paged response after its item type (`Page_haskie.document.document.Document_`),
// so the schema has no generic to alias. The envelope comes from one of them; the items stay open.
export type Page<T> = Omit<Wire<'Page_haskie.document.document.Listed_'>, 'items'> & { items: T[] }
// What an import may say about the document it creates. Everything is optional: the file name
// and the user's conversion defaults answer for whatever is left out.
type ImportOptions = Partial<Omit<Wire<'ImportRequest'>, 'staging_id' | 'path'>>

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`)
  }
  return response.status === 204 ? (undefined as T) : response.json()
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// Query string for a paged endpoint, `?k=v&…` or empty. Unset and empty values are dropped:
// an absent filter is not the same request as a filter on the empty string.
const pageQuery = (q: PageRequest, extra: Record<string, string | undefined> = {}): string => {
  const params = new URLSearchParams()
  const values: Record<string, string | number | null | undefined> = { ...q, ...extra }
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined && value !== null && value !== '') params.set(key, String(value))
  }
  const query = params.toString()
  return query ? `?${query}` : ''
}

// Read a newline-delimited JSON body frame by frame. A chunk can split a line anywhere, so the
// tail is kept until its newline arrives.
async function* ndjson<T>(path: string): AsyncGenerator<T> {
  const response = await fetch(path)
  if (!response.ok || !response.body) {
    throw new Error(`${response.status} ${response.statusText}`)
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let rest = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    rest += value
    const lines = rest.split('\n')
    rest = lines.pop() ?? ''
    for (const line of lines) if (line) yield JSON.parse(line) as T
  }
  if (rest.trim()) yield JSON.parse(rest) as T
}

const collectionPath = (name: string) => `/api/collections/${encodeURIComponent(name)}`
// A document is addressed on its own: it is imported once and shared by every collection holding it.
const documentPath = (doc: string) => `/api/documents/${encodeURIComponent(doc)}`
// One membership: the document as this collection sees it.
const memberPath = (name: string, doc: string) => `${collectionPath(name)}/documents/${encodeURIComponent(doc)}`

let optionsOnce: Promise<Options> | undefined

export const api = {
  status: () => request<Status>('/api/status'),
  init: (profile: EmbeddingProfile) => request<UserSettings>('/api/init', json('POST', { profile })),
  // The server answers a module-level constant, so one fetch per page load is enough. The promise
  // is the cache: `useOptions` reads it with React's `use`, so every view sees the same object.
  options: () => (optionsOnce ??= request<Options>('/api/options')),
  settings: () => request<UserSettings>('/api/settings'),
  saveSettings: (s: UserSettings) => request<UserSettings>('/api/settings', json('PUT', s)),

  collections: (q: PageRequest = {}) => request<Page<CollectionSummary>>(`/api/collections${pageQuery(q)}`),
  // Names alone, for a picker: one request, and the cap is the backend's largest page.
  collectionNames: () => api.collections({ page_size: MAX_PAGE_SIZE }).then((p) => p.items.map((c) => c.name)),
  createCollection: (name: string, description = '') =>
    request<CollectionInfo>('/api/collections', json('POST', { name, description })),
  describeCollection: (name: string, description: string) =>
    request<CollectionInfo>(`${collectionPath(name)}/description`, json('PUT', { description })),
  collection: (name: string) => request<CollectionInfo>(collectionPath(name)),
  deleteCollection: (name: string) => request<BulkStarted>(collectionPath(name), { method: 'DELETE' }),
  searchCollection: (name: string, q: string) => request<Hit[]>(`${collectionPath(name)}/search?q=${encodeURIComponent(q)}`),
  saveCollectionSettings: (name: string, s: CollectionSettings) =>
    request<CollectionInfo>(`${collectionPath(name)}/settings`, json('PUT', s)),
  indexCollection: (name: string) => request<BulkStarted>(`${collectionPath(name)}/index`, { method: 'POST' }),

  // The collection's members: one document row each, plus how far this collection indexed it.
  collectionDocuments: (name: string, q: PageRequest & { status?: MemberStatus } = {}) => {
    const { status, ...page } = q
    return request<Page<Member>>(`${collectionPath(name)}/documents${pageQuery(page, { status })}`)
  },
  // Adds the membership and indexes it; the document itself is already imported.
  attachDocument: (name: string, doc: string) =>
    request<BulkStarted>(`${collectionPath(name)}/documents`, json('POST', { document: doc })),
  // Drops the membership and the collection's chunks of it. The document and its embeddings stay.
  detachDocument: (name: string, doc: string) => request<void>(memberPath(name, doc), { method: 'DELETE' }),
  reindexMember: (name: string, doc: string) => request<BulkStarted>(`${memberPath(name, doc)}/index`, { method: 'POST' }),

  documents: (q: PageRequest & { status?: DocStatus } = {}) => {
    const { status, ...page } = q
    return request<Page<Document>>(`/api/documents${pageQuery(page, { status })}`)
  },
  document: (doc: string) => request<Document>(documentPath(doc)),
  // Upload step one: the bytes land in staging under an id. Nothing is imported until `importStaged`.
  stageUpload: (file: File) => {
    const body = new FormData()
    body.append('data', file)
    return request<Staged>('/api/documents/staging', { method: 'POST', body })
  },
  // Upload step two: name the staged bytes and start the import pipeline.
  importStaged: (req: ImportOptions & { staging_id: string }) => request<ImportedDocument>('/api/documents/import', json('POST', req)),
  // Import a file the server can already read, by path; the file is copied, not moved.
  importPath: (path: string, opts: ImportOptions = {}) => request<ImportedDocument>('/api/documents/import', json('POST', { path, ...opts })),
  deleteDocument: (doc: string) => request<BulkStarted>(documentPath(doc), { method: 'DELETE' }),
  // Re-run a failed or cancelled import; the backend refuses any other status.
  reimportDocument: (doc: string) => request<BulkStarted>(`${documentPath(doc)}/import`, { method: 'POST' }),
  documentCollections: (doc: string) => request<string[]>(`${documentPath(doc)}/collections`),
  documentEmbeddings: (doc: string) => request<EmbeddingEntry[]>(`${documentPath(doc)}/embeddings`),
  describeDocument: (doc: string, description: string) =>
    request<ImportedDocument>(`${documentPath(doc)}/description`, json('PUT', { description })),
  previewUrl: (doc: string) => `${documentPath(doc)}/preview`,
  sourceUrl: (doc: string) => `${documentPath(doc)}/source`,
  // Yields each frame as it arrives, so the first page shows without waiting for the last.
  markdown: (doc: string, full = false) => ndjson<Frame>(`${documentPath(doc)}/markdown${full ? '?full=true' : ''}`),

  // One section of the Operations view. The cursor is an opaque offset rather than a keyset,
  // because the listing reads the run history, which has one fixed ordering (newest first) and no
  // sort of its own. A cursor belongs to the kind that issued it.
  operations: (kind: OperationKind, q: PageRequest & { collection?: string } = {}) =>
    request<Page<Operation>>(`/api/operations${pageQuery({ cursor: q.cursor, page_size: q.page_size }, { kind, collection: q.collection })}`),
  operationKinds: () => request<OperationKindSummary[]>('/api/operations/kinds'),
  activity: () => request<Activity>('/api/operations/activity'),
  // One job of an operation: its micro-batches.
  jobTasks: (jobId: string) => request<Task[]>(`/api/jobs/${jobId}/tasks`),
  operationProgress: (operationId: string) => request<OperationProgress>(`/api/operations/${operationId}/progress`),
  cancelOperation: (operationId: string) => request<void>(`/api/operations/${operationId}`, { method: 'DELETE' }),

  // Full-text search across every collection, for Explore's "all collections" scope: there is no
  // session and no one collection to answer the query, so the paged endpoint stands in for one.
  // One page of passages is what Explore shows.
  searchText: (q: string) => request<Page<Hit>>(`/api/search/text${pageQuery({ page_size: 50 }, { q })}`),
  // Which documents to read for a query, rather than which passages answer it.
  searchDocuments: (q: string, collections?: string[], limit?: number) =>
    request<DocumentMatch[]>(`/api/search/documents${pageQuery({}, { q, collections: collections?.join(','), limit: limit?.toString() })}`),
  // The passages behind one row of `searchDocuments`: the same scan, kept to that document.
  documentPassages: (doc: string, q: string, collections?: string[]) =>
    request<Hit[]>(`/api/search/documents/${encodeURIComponent(doc)}${pageQuery({}, { q, collections: collections?.join(',') })}`),

  sessions: () => request<SessionSummary[]>('/api/sessions'),
  sessionHistory: (id: string) => request<SessionEvent[]>(`/api/sessions/${encodeURIComponent(id)}/history`),
  searchTrend: (days: number) => request<SearchAt[]>(`/api/insights/searches${pageQuery({}, { days: String(days) })}`),
  chunkTrend: (days: number) => request<ChunksAt[]>(`/api/insights/chunks${pageQuery({}, { days: String(days) })}`),
  saveSession: (id: string, collections: string[]) =>
    request<string[]>(`/api/sessions/${encodeURIComponent(id)}`, json('PUT', { collections })),
  search: (sessionId: string, q: string) =>
    request<Hit[]>(`/api/search/explore?granularity=chunk&session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(q)}`),
}
