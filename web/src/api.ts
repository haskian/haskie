export type Parser = 'anydoc' | 'plain'
export type Chunker = 'markdown' | 'text'
export type EmbeddingProfile = 'none' | 'compact' | 'quality' | 'multilingual'

export type Accelerator = 'auto' | 'cpu'
export interface EmbeddingModel {
  name: string
  dims: number
  accelerator: Accelerator
}
// How one collection splits a document's markdown into chunks. These three values key the
// embedding cache, so two collections that agree on them share one set of embeddings.
export interface ChunkSettings {
  chunker: Chunker
  chunk_size: number
  chunk_overlap: number
}
// The user-level defaults: how a document is converted when nothing is said at import (`parser`,
// `skip_ocr_pages`), and how a collection chunks it when it overrides nothing.
export interface ConversionSettings extends ChunkSettings {
  parser: Parser
  skip_ocr_pages: boolean
}
// A collection's chunking overrides; null means "use the user default".
export type ChunkOverrides = { [K in keyof ChunkSettings]: ChunkSettings[K] | null }
// Everything a collection may override. Conversion runs once per document at import, so `parser`
// and `skip_ocr_pages` are not here: they belong to the document.
export interface CollectionSettings extends ChunkOverrides {
  search: SearchOverrides
}
export type SearchMode = 'hybrid' | 'vector' | 'fts'
export type Fusion = 'rrf' | 'linear'
export type Reranker = 'none' | 'cross-encoder'
export interface SearchSettings {
  limit: number
  candidates: number
  mode: SearchMode
  fusion: Fusion
  rrf_k: number
  vector_weight: number
  bm25_weight: number
  nprobes: number
  refine_factor: number
  reranker: Reranker
  reranker_model: string
}
export interface FieldDoc {
  title: string
  description: string
}
export type SearchOverrides = { [K in keyof SearchSettings]: SearchSettings[K] | null }
// How the convert -> embed -> index pipeline runs, for the whole process.
export interface PipelineSettings {
  cpu_budget: number
  converting_weight: number
  embedding_weight: number
  indexing_weight: number
  document_parallelism: number
  batch_pages: number
  index_group_parts: number
  task_timeout_seconds: number
  maintenance_docs: number
  maintenance_idle_seconds: number
  ann_min_rows: number
  preview_workers: number
  accelerator: Accelerator
}
// How long history is kept: job history in two places, plus the audit trail.
export interface RetentionSettings {
  job_days: number
  job_live_hours: number
  audit_days: number
}
export interface UserSettings {
  embedding: EmbeddingProfile
  conversion: ConversionSettings
  pipeline: PipelineSettings
  search: SearchSettings
  retention: RetentionSettings
}
export interface Options {
  parsers: Parser[]
  chunkers: Chunker[]
  accelerators: Accelerator[]
  search_modes: SearchMode[]
  fusions: Fusion[]
  rerankers: Reranker[]
  reranker_models: string[]
  docs: Record<string, FieldDoc>
  embedding_profiles: Record<EmbeddingProfile, EmbeddingModel | null>
}
export interface Preview {
  kind: 'pdf' | 'image' | 'text' | 'html'
  truncated: boolean
  pages: number | null
  ocr_pages: number[]
}
// A document's own lifecycle: imported once, then held by any number of collections.
export type DocStatus = 'queued' | 'converting' | 'embedding' | 'imported' | 'error' | 'cancelled' | 'deleting'
export const DOCUMENT_STATUSES: readonly DocStatus[] = ['queued', 'converting', 'embedding', 'imported', 'error', 'cancelled', 'deleting']
// in the import pipeline right now: the states a poll waits on
export const ACTIVE_DOCUMENT_STATUSES: readonly DocStatus[] = ['queued', 'converting', 'embedding']
// How far one collection got writing one of its documents into its own table.
export type MemberStatus = 'pending' | 'indexing' | 'indexed' | 'error' | 'cancelled'
export const MEMBER_STATUSES: readonly MemberStatus[] = ['pending', 'indexing', 'indexed', 'error', 'cancelled']
export const ACTIVE_MEMBER_STATUSES: readonly MemberStatus[] = ['pending', 'indexing']
export const ACTIVE_JOB_STATUSES: ReadonlySet<string> = new Set(['ENQUEUED', 'PENDING'])
export interface Document {
  name: string
  suffix: string // of the original file, lower-case, with the dot: ".pdf"
  size: number
  status: DocStatus
  error: string | null
  preview: Preview | null
  parser: Parser
  skip_ocr_pages: boolean
  created_at: number // unix seconds
  updated_at: number
  description: string // what the document is, in the importer's words
}
// An upload waiting in the staging area: bytes on the server, nothing in the database yet. It
// becomes a document only when `importStaged` names it.
export interface Staged {
  staging_id: string
  filename: string
  size: number
}
// One document as a member of one collection: the document row, and how far this collection got
// writing it into its table.
export interface Member {
  document: Document
  status: MemberStatus
  error: string | null
  added_at: number
  updated_at: number
}
// How a collection's memberships are spread over the lifecycle, counted by the backend: the
// member listing is paged, so the rows on screen are never the whole collection.
export interface DocumentCounts {
  total: number
  indexed: number
  active: number
  error: number
  by_status: Record<string, number>
}
export interface CollectionSummary {
  name: string
  counts: DocumentCounts
  created_at: number
  description: string
}
// The collection's LanceDB table as it is right now, plus how its maintenance stands. Null until
// the collection has a table (see the backend's IndexStatus).
export interface IndexStatus {
  num_rows: number
  num_fragments: number
  num_small_fragments: number
  has_fts_index: boolean
  has_vector_index: boolean
  unindexed_rows: number
  vector_index_rows: number
  last_maintained_at: number | null // unix seconds
  pending_docs: number
}
export interface CollectionInfo {
  name: string
  settings: CollectionSettings
  effective: ChunkSettings // what the overrides above resolve to
  search: SearchSettings
  description: string
  counts: DocumentCounts
  index_outdated: boolean
  index: IndexStatus | null
}
// One row of the embedding cache: a set of vectors on disk, keyed by the document plus every
// setting that decided them. Two collections with the same key share this one entry.
export interface EmbeddingEntry {
  id: string
  document: string
  urn: string
  model: string
  chunk_size: number
  chunk_overlap: number
  chunker: string
  chunk_version: number
  parser: string
  skip_ocr_pages: boolean
  rows: number
  bytes: number
  created_at: number
}
// Work the backend only accepts (202) and runs in the background: "index all", the deletion of a
// collection or a document, attaching a document. `job_id` is what a poll of `jobProgress` follows.
export interface BulkStarted {
  job_id: string
}
export type BulkKind = 'index_collection' | 'delete_collection' | 'delete_document'
export interface BulkProgress {
  done: number
  skipped: number
  total: number
  last: string | null // last document of the page queued; null once the collection is exhausted
}
export interface BulkJob {
  id: string
  kind: BulkKind
  collection: string | null // null for a document deletion, which belongs to no collection
  status: string // DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED
  progress: BulkProgress | null // null for a deletion, which has no pages
  error: string | null
}
// Every kind of background work the backend runs, in the order the Jobs view shows them.
export type JobKind = 'document' | 'collection' | 'download' | 'maintenance' | 'archive'
// One job of any kind: what every kind has in common, plus the numbers only that kind has in
// `detail` (a document: tasks_done/tasks_running/tasks_total; a collection job: done/skipped/total;
// a model download: warm, which says the model is loaded in the backend process).
export interface JobRow {
  id: string
  kind: JobKind
  title: string // human text: "import doc", "collection / doc", "index collection X"
  status: string // DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED
  created_at: number
  updated_at: number
  error: string | null
  archived: boolean // documents only: read from a day partition instead of the DBOS tables
  detail: Record<string, number | string | boolean | null>
}
// One section of the Jobs view, with how many of its jobs are running right now.
export interface JobKindSummary {
  kind: JobKind
  label: string
  active: number
}
export type Stage = 'convert' | 'embed' | 'index'
// The nav indicator: coarse jobs (job.* queues) and the tasks they are made of (task.* queues).
export interface QueueActivity {
  queued: number // waiting for a slot, or for a debounce to expire
  running: number
}
export interface Activity {
  jobs: QueueActivity
  tasks: QueueActivity
}
export interface Task {
  id: string
  stage: Stage
  seq: number
  page_start: number
  page_end: number
  status: string
  result: number | null
  error: string | null
}
export interface Heading {
  level: number
  text: string
  offset: number
}
// The viewer is streamed as NDJSON: one `head`, then one `page` per page of rendered HTML. The
// HTML comes from the server (pyromark) with raw HTML already removed, which is why the pane can
// insert it; see `haskie/render.py`.
export interface Head {
  kind: 'head'
  toc: Heading[]
  preview: Preview | null
  pages: number
}
export interface Page_ {
  kind: 'page'
  number: number | null
  html: string
}
export type Frame = Head | Page_
export interface Hit {
  collection: string // the collection whose table matched; the document itself belongs to none
  doc: string
  home: string
  source_path: string
  markdown_path: string
  part: number
  chunk_id: number
  line_start: number
  line_end: number
  char_start: number
  char_end: number
  page_start: number | null
  page_end: number | null
  parents: string[]
  heading: string
  header: string
  location: string
  text: string
  score: number
  // Absolute, for anything outside the app that opens or greps the file; `line_start`/`line_end`
  // are line numbers in `markdown_file`.
  source_file: string
  markdown_file: string
}
// One document a query matched, and the best evidence that it did: the answer to "which
// documents should I read", as opposed to `Hit`, which answers "which passage says so".
export interface DocumentMatch {
  collection: string
  doc: string
  score: number
  chunks: number
  description: string
  heading: string
  location: string
  text: string
  source_file: string
  markdown_file: string
  line_start: number
  line_end: number
}
export interface ModelStatus {
  kind: 'embedding' | 'reranker'
  name: string
  state: 'pending' | 'loading' | 'ready' | 'error'
  error: string | null
}
export interface Status {
  initialized: boolean
  home: string
  embedding: EmbeddingModel | null
  device: string
  models: ModelStatus[]
  settings_error: string | null // stored settings unreadable; defaults are in use
}

export type Order = 'asc' | 'desc'
export const DEFAULT_PAGE_SIZE = 100
export const MAX_PAGE_SIZE = 1000 // the backend's cap; a bigger page_size is rejected
// Query arguments of one page; `cursor` is opaque and only ever comes from a previous `Page`.
export interface PageRequest {
  cursor?: string | null
  page_size?: number
  sort?: string
  order?: Order
}
export interface Page<T> {
  items: T[]
  next_cursor: string | null // null on the last page
  total: number | null
}
// What an import may say about the document it creates. Everything is optional: the file name
// and the user's conversion defaults answer for whatever is left out.
export interface ImportOptions {
  name?: string
  description?: string
  parser?: Parser
  skip_ocr_pages?: boolean
}

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
export const pageQuery = (q: PageRequest, extra: Record<string, string | undefined> = {}): string => {
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
  // The server answers a module-level constant, so one fetch per page load is enough.
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
  searchCollection: (name: string, q: string, limit?: number) =>
    request<Hit[]>(`${collectionPath(name)}/search?q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`),
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
  importStaged: (req: ImportOptions & { staging_id: string }) => request<Document>('/api/documents/import', json('POST', req)),
  // Import a file the server can already read, by path; the file is copied, not moved.
  importPath: (path: string, opts: ImportOptions = {}) => request<Document>('/api/documents/import', json('POST', { path, ...opts })),
  deleteDocument: (doc: string) => request<BulkStarted>(documentPath(doc), { method: 'DELETE' }),
  // Re-run a failed or cancelled import; the backend refuses any other status.
  reimportDocument: (doc: string) => request<BulkStarted>(`${documentPath(doc)}/import`, { method: 'POST' }),
  documentCollections: (doc: string) => request<string[]>(`${documentPath(doc)}/collections`),
  documentEmbeddings: (doc: string) => request<EmbeddingEntry[]>(`${documentPath(doc)}/embeddings`),
  describeDocument: (doc: string, description: string) =>
    request<Document>(`${documentPath(doc)}/description`, json('PUT', { description })),
  previewUrl: (doc: string) => `${documentPath(doc)}/preview`,
  sourceUrl: (doc: string) => `${documentPath(doc)}/source`,
  // Yields each frame as it arrives, so the first page shows without waiting for the last.
  markdown: (doc: string, full = false) => ndjson<Frame>(`${documentPath(doc)}/markdown${full ? '?full=true' : ''}`),

  // One section of the Jobs view. The cursor is an opaque offset rather than a keyset, because
  // the listing reads DBOS's workflow history, which has one fixed ordering (newest first) and no
  // sort of its own. A cursor belongs to the kind that issued it.
  jobsByKind: (kind: JobKind, q: PageRequest & { collection?: string } = {}) =>
    request<Page<JobRow>>(`/api/jobs/by-kind${pageQuery({ cursor: q.cursor, page_size: q.page_size }, { kind, collection: q.collection })}`),
  jobKinds: () => request<JobKindSummary[]>('/api/jobs/kinds'),
  activity: () => request<Activity>('/api/jobs/activity'),
  jobTasks: (jobId: string) => request<Task[]>(`/api/jobs/${jobId}/tasks`),
  jobProgress: (jobId: string) => request<BulkJob>(`/api/jobs/${jobId}/progress`),
  deleteJob: (jobId: string) => request<void>(`/api/jobs/${jobId}`, { method: 'DELETE' }),

  // Which documents to read for a query, rather than which passages answer it.
  searchDocuments: (q: string, collections?: string[], limit?: number) =>
    request<DocumentMatch[]>(`/api/search/documents${pageQuery({}, { q, collections: collections?.join(','), limit: limit?.toString() })}`),

  sessions: () => request<Record<string, string[]>>('/api/sessions'),
  saveSession: (id: string, collections: string[]) =>
    request<string[]>(`/api/sessions/${encodeURIComponent(id)}`, json('PUT', { collections })),
  search: (sessionId: string, q: string, limit?: number) =>
    request<Hit[]>(`/api/search?session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`),
}
