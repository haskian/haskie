import type { BulkKind, Operation, OperationKind, RunStatus, Stage, Task } from '../../api'
import type { JobBar, JobState } from '../../ui'

// A document operation runs some of four stages: an import converts, embeds and describes, an
// index embeds, describes and writes, where its embedding is missing. The stages cost different
// amounts of time. Embedding dominates, so its bar is the widest one.
export interface StageDef {
  stage: Stage
  label: string
  weight: number
}
const DOCUMENT_STAGES: Record<Stage, Omit<StageDef, 'stage'>> = {
  convert: { label: 'Convert', weight: 1 },
  embed: { label: 'Embed', weight: 2.2 },
  describe: { label: 'Describe', weight: 1.4 },
  index: { label: 'Index', weight: 1 },
}

// The word on the tag and the one bar that stands for an operation with no jobs of its own.
const SINGLE: Record<Exclude<OperationKind, WholeKind | 'document'>, { tag: string; label: string }> = {
  download: { tag: 'Download', label: 'Download' },
  maintenance: { tag: 'Maintain', label: 'Maintenance' },
}
// The collection and backup kinds are whole-thing operations: an index queues one document at a
// time, a backup archives one file at a time, a restore, the deletes and a description have
// nothing to count. `detail.bulk` says which; a row without it is read as an index.
type WholeKind = 'collection' | 'backup'
const isWhole = (kind: OperationKind): kind is WholeKind => kind === 'collection' || kind === 'backup'
const BULK: Record<BulkKind, { tag: string; label: string }> = {
  index_collection: { tag: 'Index', label: 'Queue' },
  delete_collection: { tag: 'Delete', label: 'Delete' },
  delete_document: { tag: 'Delete', label: 'Delete' },
  summarize_document: { tag: 'Describe', label: 'Describe' },
  summarize_collection: { tag: 'Describe', label: 'Describe' },
  create_backup: { tag: 'Backup', label: 'Archive' },
  restore_backup: { tag: 'Restore', label: 'Restore' },
}
const isBulkKind = (kind: unknown): kind is BulkKind => typeof kind === 'string' && kind in BULK
export const bulkKind = (operation: Operation): BulkKind => (isBulkKind(operation.detail.bulk) ? operation.detail.bulk : 'index_collection')

/** The word on a row's tag: for a document, which operation it is; for the rest, the kind. */
export function tagOf(operation: Operation): string {
  if (isWhole(operation.kind)) return BULK[bulkKind(operation)].tag
  if (operation.kind !== 'document') return SINGLE[operation.kind].tag
  const stages = operation.jobs.map((job) => job.stage)
  if (stages.includes('convert')) return 'Import'
  return stages.includes('index') ? 'Index' : 'Embed'
}

/** The jobs an operation's strip and task columns show, in the pipeline's order, which is the
 *  order the backend lists them in. */
export function jobDefs(operation: Operation): StageDef[] {
  return operation.jobs.map((job) => ({ stage: job.stage, ...DOCUMENT_STAGES[job.stage] }))
}

/** One of an operation's own counters, or zero where it never reported it. */
export const count = (operation: Operation, key: string): number => (typeof operation.detail[key] === 'number' ? operation.detail[key] : 0)

/** How an operation or one of its jobs stands, read from its status alone. `active` is the
 *  backend's list of run statuses still on their way (`Options.active_run_statuses`). */
export function runState(status: RunStatus, active: readonly RunStatus[]): JobState {
  if (status === 'SUCCESS') return 'done'
  if (status === 'ERROR') return 'error'
  return active.includes(status) ? 'active' : 'todo'
}

function stateFromTasks(rows: Task[], done: number, active: readonly RunStatus[]): JobState {
  if (rows.length > 0 && done === rows.length) return 'done'
  if (rows.some((task) => active.includes(task.status))) return 'active'
  if (rows.some((task) => task.status === 'ERROR')) return 'error'
  return 'todo'
}

/**
 * The bars in an operation's summary. `tasks` is null until they are fetched (and empty for an
 * operation that ran none), which is why a document also reads its coarse counters from `detail`.
 */
export function jobsFor(operation: Operation, tasks: Task[] | null, active: readonly RunStatus[]): JobBar[] {
  if (operation.kind === 'document') return documentJobs(operation, tasks ?? [], active)
  const state = runState(operation.status, active)
  // An operation with no jobs of its own is one run, so the bar lasted as long as the operation.
  const seconds = state === 'done' || state === 'error' ? operation.updated_at - operation.created_at : undefined
  if (isWhole(operation.kind)) {
    return [{ label: BULK[bulkKind(operation)].label, state, seconds, done: count(operation, 'done'), total: count(operation, 'total') }]
  }
  // one task is the whole operation: 1/1 once it is done, so the row reads like the others
  const row: JobBar = { label: SINGLE[operation.kind].label, state, seconds, done: state === 'done' ? 1 : 0, total: 1 }
  if (operation.kind === 'download') row.note = operation.detail.warm ? 'loaded' : 'not loaded'
  return [row]
}

/**
 * One bar per job. Its tasks decide once they are fetched; until then, and for a job that ran none
 * (an embedding the cache already held), the job's own status and counters do.
 */
function documentJobs(operation: Operation, tasks: Task[], active: readonly RunStatus[]): JobBar[] {
  return operation.jobs.map((job) => {
    const { label, weight } = DOCUMENT_STAGES[job.stage]
    const rows = tasks.filter((task) => task.stage === job.stage)
    const seconds = job.seconds ?? undefined
    if (rows.length === 0) {
      const state = runState(job.status, active)
      const note = job.stage === 'embed' && state === 'done' && job.tasks_total === 0 ? 'cached' : undefined
      return { label, weight, done: job.tasks_done, total: job.tasks_total, state, note, seconds }
    }
    const done = rows.filter((task) => task.status === 'SUCCESS').length
    return { label, weight, done, total: rows.length, state: stateFromTasks(rows, done, active), seconds }
  })
}

const TASKS_AT_EACH_END = 3 // the first and last batches are listed, the rest counted: a document can have hundreds

/** The first and last few of a list, and how many sit between them. Six or fewer are all head. */
export function endsOf<T>(rows: T[], each = TASKS_AT_EACH_END): { head: T[]; hidden: number; tail: T[] } {
  if (rows.length <= each * 2) return { head: rows, hidden: 0, tail: [] }
  return { head: rows.slice(0, each), hidden: rows.length - each * 2, tail: rows.slice(-each) }
}

/** What one task covered: pages for a conversion, a part for an embed, sections for a
 *  description, parts for an index write, or the document or collection a task names. */
export function taskText(task: Task): string {
  if (task.name !== null) return task.name
  if (task.stage === 'convert') return `pages ${task.page_start + 1}–${task.page_end}`
  if (task.stage === 'embed') return `part ${task.page_start}`
  if (task.stage === 'describe') return `sections ${task.page_start + 1}–${task.page_end}`
  return `parts ${task.page_start}–${task.page_end}`
}

export function taskState(task: Task): 'done' | 'error' | 'todo' {
  if (task.status === 'SUCCESS') return 'done'
  return task.status === 'ERROR' ? 'error' : 'todo'
}

const RESULT_UNITS: Record<Stage, string> = { convert: 'OCR pages', embed: 'chunks', describe: 'sections', index: 'chunks' }

/** What a job produced, summed over its tasks: OCR pages for a conversion, sections for a
 *  description, chunks otherwise. */
export function stageInfo(stage: Stage, rows: Task[]): string {
  const results = rows.filter((task) => task.result !== null)
  if (results.length === 0) return ''
  const total = results.reduce((sum, task) => sum + (task.result ?? 0), 0)
  return `${total} ${RESULT_UNITS[stage]}`
}
