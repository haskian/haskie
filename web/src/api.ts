export type Parser = 'anydoc' | 'plain'
export type Chunker = 'markdown' | 'text'
export type EmbeddingProfile = 'none' | 'compact' | 'quality' | 'multilingual'

export type Accelerator = 'auto' | 'cpu'
export interface EmbeddingModel {
  name: string
  dims: number
  accelerator: Accelerator
}
export interface ConversionSettings {
  parser: Parser
  chunker: Chunker
  chunk_size: number
  chunk_overlap: number
  skip_ocr_pages: boolean
}
export type ConversionOverrides = { [K in keyof ConversionSettings]: ConversionSettings[K] | null }
export interface LibrarySettings extends ConversionOverrides {
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
export type DocStatus = 'uploaded' | 'queued' | 'converting' | 'embedding' | 'indexing' | 'indexed' | 'error' | 'cancelled'
export const DOCUMENT_STATUSES: readonly DocStatus[] = ['uploaded', 'queued', 'converting', 'embedding', 'indexing', 'indexed', 'error', 'cancelled']
export const ACTIVE_DOCUMENT_STATUSES: readonly DocStatus[] = ['queued', 'converting', 'embedding', 'indexing']
export const ACTIVE_JOB_STATUSES: ReadonlySet<string> = new Set(['ENQUEUED', 'PENDING'])
export interface Document {
  name: string
  size: number
  status: DocStatus
  error: string | null
  preview: Preview | null
  created_at: number // unix seconds
  updated_at: number
  description: string // what the document is, in the uploader's words
}
// How a library's documents are spread over the lifecycle, counted by the backend: the document
// listing is paged, so the rows on screen are never the whole library.
export interface DocumentCounts {
  total: number
  indexed: number
  active: number
  error: number
  by_status: Record<string, number>
}
export interface LibrarySummary {
  name: string
  counts: DocumentCounts
  created_at: number
  description: string
}
// The library's LanceDB table as it is right now, plus how its maintenance stands. Null until
// the library has a table (see the backend's IndexStatus).
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
export interface LibraryInfo {
  name: string
  settings: LibrarySettings
  effective: ConversionSettings
  search: SearchSettings
  description: string
  counts: DocumentCounts
  index_outdated: boolean
  index: IndexStatus | null
}
// Whole-library work the backend only accepts (202) and runs in the background: "index all" and
// the deletion of a library. `job_id` is what a poll of `jobProgress` follows.
export interface BulkStarted {
  job_id: string
}
export type BulkKind = 'index_library' | 'delete_library'
export interface BulkProgress {
  done: number
  skipped: number
  total: number
  last: string | null // last document of the page queued; null once the library is exhausted
}
export interface BulkJob {
  id: string
  kind: BulkKind
  library: string
  status: string // DBOS: ENQUEUED | PENDING | SUCCESS | ERROR | CANCELLED
  progress: BulkProgress | null // null for a deletion, which has no pages
  error: string | null
}
// Every kind of background work the backend runs, in the order the Jobs view shows them.
export type JobKind = 'document' | 'library' | 'download' | 'maintenance' | 'archive'
// One job of any kind: what every kind has in common, plus the numbers only that kind has in
// `detail` (a document: tasks_done/tasks_running/tasks_total; a bulk index: done/skipped/total;
// a model download: warm, which says the model is loaded in the backend process).
export interface JobRow {
  id: string
  kind: JobKind
  title: string // human text: "library / doc", "index library X", "download reranker Y"
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
  library: string
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
  library: string
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

const libraryPath = (name: string) => `/api/libraries/${encodeURIComponent(name)}`
const documentPath = (name: string, doc: string) => `${libraryPath(name)}/documents/${encodeURIComponent(doc)}`

let optionsOnce: Promise<Options> | undefined

export const api = {
  status: () => request<Status>('/api/status'),
  init: (profile: EmbeddingProfile) => request<UserSettings>('/api/init', json('POST', { profile })),
  // The server answers a module-level constant, so one fetch per page load is enough.
  options: () => (optionsOnce ??= request<Options>('/api/options')),
  settings: () => request<UserSettings>('/api/settings'),
  saveSettings: (s: UserSettings) => request<UserSettings>('/api/settings', json('PUT', s)),

  libraries: (q: PageRequest = {}) => request<Page<LibrarySummary>>(`/api/libraries${pageQuery(q)}`),
  // Names alone, for a picker: one request, and the cap is the backend's largest page.
  libraryNames: () => api.libraries({ page_size: MAX_PAGE_SIZE }).then((p) => p.items.map((l) => l.name)),
  createLibrary: (name: string, description = '') =>
    request<LibraryInfo>('/api/libraries', json('POST', { name, description })),
  describeLibrary: (name: string, description: string) =>
    request<LibraryInfo>(`${libraryPath(name)}/description`, json('PUT', { description })),
  library: (name: string) => request<LibraryInfo>(libraryPath(name)),
  documents: (name: string, q: PageRequest & { status?: DocStatus } = {}) => {
    const { status, ...page } = q
    return request<Page<Document>>(`${libraryPath(name)}/documents${pageQuery(page, { status })}`)
  },
  deleteLibrary: (name: string) => request<BulkStarted>(libraryPath(name), { method: 'DELETE' }),
  searchLibrary: (name: string, q: string, limit?: number) =>
    request<Hit[]>(`${libraryPath(name)}/search?q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`),
  saveLibrarySettings: (name: string, s: LibrarySettings) =>
    request<LibraryInfo>(`${libraryPath(name)}/settings`, json('PUT', s)),
  indexLibrary: (name: string) => request<BulkStarted>(`${libraryPath(name)}/index`, { method: 'POST' }),
  // One section of the Jobs view. The cursor is an opaque offset rather than a keyset, because
  // the listing reads DBOS's workflow history, which has one fixed ordering (newest first) and no
  // sort of its own. A cursor belongs to the kind that issued it.
  jobsByKind: (kind: JobKind, q: PageRequest & { library?: string } = {}) =>
    request<Page<JobRow>>(`/api/jobs/by-kind${pageQuery({ cursor: q.cursor, page_size: q.page_size }, { kind, library: q.library })}`),
  jobKinds: () => request<JobKindSummary[]>('/api/jobs/kinds'),
  activity: () => request<Activity>('/api/jobs/activity'),
  jobTasks: (jobId: string) => request<Task[]>(`/api/jobs/${jobId}/tasks`),
  jobProgress: (jobId: string) => request<BulkJob>(`/api/jobs/${jobId}/progress`),
  deleteJob: (jobId: string) => request<void>(`/api/jobs/${jobId}`, { method: 'DELETE' }),

  upload: (name: string, f: File, opts: { rename_to?: string; description?: string } = {}) => {
    const body = new FormData()
    body.append('data', f)
    const query = pageQuery({}, { rename_to: opts.rename_to, description: opts.description })
    return request<Document>(`${libraryPath(name)}/documents${query}`, { method: 'POST', body })
  },
  describeDocument: (name: string, doc: string, description: string) =>
    request<Document>(`${documentPath(name, doc)}/description`, json('PUT', { description })),
  indexDocument: (name: string, doc: string) => request<BulkStarted>(`${documentPath(name, doc)}/index`, { method: 'POST' }),
  deleteDocument: (name: string, doc: string) => request<void>(documentPath(name, doc), { method: 'DELETE' }),
  previewUrl: (name: string, doc: string) => `${documentPath(name, doc)}/preview`,
  sourceUrl: (name: string, doc: string) => `${documentPath(name, doc)}/source`,
  // Yields each frame as it arrives, so the first page shows without waiting for the last.
  markdown: (name: string, doc: string, full = false) =>
    ndjson<Frame>(`${documentPath(name, doc)}/markdown${full ? '?full=true' : ''}`),

  // Which documents to read for a query, rather than which passages answer it.
  searchDocuments: (q: string, libraries?: string[], limit?: number) =>
    request<DocumentMatch[]>(`/api/search/documents${pageQuery({}, { q, libraries: libraries?.join(','), limit: limit?.toString() })}`),

  sessions: () => request<Record<string, string[]>>('/api/sessions'),
  saveSession: (id: string, libraries: string[]) =>
    request<string[]>(`/api/sessions/${encodeURIComponent(id)}`, json('PUT', { libraries })),
  search: (sessionId: string, q: string, limit?: number) =>
    request<Hit[]>(`/api/search?session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(q)}${limit ? `&limit=${limit}` : ''}`),
}
