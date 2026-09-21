import { ACTIVE_JOB_STATUSES, type JobKind, type JobRow, type Stage, type Task, type WorkflowStatus } from '../../api'
import type { StageRow, StageState } from '../../ui'

// A document runs the same three stages every time, and they cost different amounts of time:
// embedding dominates, so its bar is the widest one.
export interface StageDef {
  stage: Stage
  label: string
  weight: number
}
const DOCUMENT_STAGES: Record<Stage, Omit<StageDef, 'stage'>> = {
  convert: { label: 'Convert', weight: 1 },
  embed: { label: 'Embed', weight: 2.2 },
  index: { label: 'Index', weight: 1 },
}

// The word on the tag and the one stage that stands for a job with no stages of its own.
const SINGLE: Record<Exclude<JobKind, 'document' | 'collection'>, { tag: string; stage: string }> = {
  download: { tag: 'Download', stage: 'Download' },
  maintenance: { tag: 'Maintain', stage: 'Maintenance' },
}
// The collection kind is three bulk jobs: an index queues one document at a time, the deletes
// have nothing to count. `detail.bulk` says which; a row without it is read as an index.
type BulkKind = 'index_collection' | 'delete_collection' | 'delete_document'
const BULK: Record<BulkKind, { tag: string; stage: string }> = {
  index_collection: { tag: 'Index', stage: 'Queue' },
  delete_collection: { tag: 'Delete', stage: 'Delete' },
  delete_document: { tag: 'Delete', stage: 'Delete' },
}
const isBulkKind = (kind: unknown): kind is BulkKind => typeof kind === 'string' && kind in BULK
export const bulkKind = (job: JobRow): BulkKind => (isBulkKind(job.detail.bulk) ? job.detail.bulk : 'index_collection')

/** The word on a row's tag: for a document, which operation it is; for the rest, the kind. */
export function tagOf(job: JobRow): string {
  if (job.kind === 'collection') return BULK[bulkKind(job)].tag
  if (job.kind !== 'document') return SINGLE[job.kind].tag
  const stages = job.stages.map((stage) => stage.stage)
  if (stages.includes('convert')) return 'Import'
  return stages.includes('index') ? 'Index' : 'Embed'
}

/** The stages an operation's strip and task columns show: the jobs it is made of, in the
 *  pipeline's order, which is the order the backend lists them in. */
export function stageDefs(job: JobRow): StageDef[] {
  return job.stages.map((one) => ({ stage: one.stage, ...DOCUMENT_STAGES[one.stage] }))
}

/** One of a job's own counters, or zero where the job never reported it. */
export const count = (job: JobRow, key: string): number => (typeof job.detail[key] === 'number' ? job.detail[key] : 0)

/** How a job or one of its stage jobs stands, read from its status alone. */
export function jobState(status: WorkflowStatus): StageState {
  if (status === 'SUCCESS') return 'done'
  if (status === 'ERROR') return 'error'
  return ACTIVE_JOB_STATUSES.has(status) ? 'active' : 'todo'
}

function taskStageState(rows: Task[], done: number): StageState {
  if (rows.length > 0 && done === rows.length) return 'done'
  if (rows.some((task) => ACTIVE_JOB_STATUSES.has(task.status))) return 'active'
  if (rows.some((task) => task.status === 'ERROR')) return 'error'
  return 'todo'
}

/**
 * The bars in a job's summary. `tasks` is null until they are fetched (and empty for a job that
 * ran none), which is why a document job also reads its coarse counters from `detail`.
 */
export function stagesFor(job: JobRow, tasks: Task[] | null): StageRow[] {
  if (job.kind === 'document') return documentStages(job, tasks ?? [])
  const state = jobState(job.status)
  // A one-stage job is its whole workflow, so the stage ran as long as the job did.
  const seconds = state === 'done' || state === 'error' ? job.updated_at - job.created_at : undefined
  if (job.kind === 'collection') {
    const skipped = count(job, 'skipped')
    return [
      {
        label: BULK[bulkKind(job)].stage,
        state,
        seconds,
        done: count(job, 'done'),
        total: count(job, 'total'),
        // Skipped documents are the only thing the queue has to say beyond its counts.
        note: skipped > 0 ? `${skipped} skipped` : undefined,
      },
    ]
  }
  // one task is the whole job: 1/1 once it is done, so the row reads like the others
  const row: StageRow = { label: SINGLE[job.kind].stage, state, seconds, done: state === 'done' ? 1 : 0, total: 1 }
  if (job.kind === 'download') row.note = job.detail.warm ? 'loaded' : 'not loaded'
  return [row]
}

/**
 * One bar per stage job. Its batches decide once they are fetched; until then, and for a stage
 * that ran none (an embedding the cache already held), the stage job's own status and counters do.
 */
function documentStages(job: JobRow, tasks: Task[]): StageRow[] {
  return job.stages.map((stage) => {
    const { label, weight } = DOCUMENT_STAGES[stage.stage]
    const rows = tasks.filter((task) => task.stage === stage.stage)
    const seconds = stage.seconds ?? undefined
    if (rows.length === 0) {
      const state = jobState(stage.status)
      const note = stage.stage === 'embed' && state === 'done' && stage.tasks_total === 0 ? 'cached' : undefined
      return { label, weight, done: stage.tasks_done, total: stage.tasks_total, state, note, seconds }
    }
    const done = rows.filter((task) => task.status === 'SUCCESS').length
    return { label, weight, done, total: rows.length, state: taskStageState(rows, done), seconds }
  })
}

const TASKS_AT_EACH_END = 3 // the first and last batches are listed, the rest counted: a document can have hundreds

/** The first and last few of a list, and how many sit between them. Six or fewer are all head. */
export function endsOf<T>(rows: T[], each = TASKS_AT_EACH_END): { head: T[]; hidden: number; tail: T[] } {
  if (rows.length <= each * 2) return { head: rows, hidden: 0, tail: [] }
  return { head: rows.slice(0, each), hidden: rows.length - each * 2, tail: rows.slice(-each) }
}

/** What one micro-batch covered: pages for a conversion, parts for an embed or an index write. */
export function taskText(task: Task): string {
  if (task.stage === 'convert') return `pages ${task.page_start + 1}–${task.page_end}`
  if (task.stage === 'embed') return `part ${task.page_start}`
  return `parts ${task.page_start}–${task.page_end}`
}

export function taskState(task: Task): 'done' | 'error' | 'todo' {
  if (task.status === 'SUCCESS') return 'done'
  return task.status === 'ERROR' ? 'error' : 'todo'
}

/** What a stage produced, summed over its batches: OCR pages for a conversion, chunks otherwise. */
export function stageInfo(stage: Stage, rows: Task[]): string {
  const results = rows.filter((task) => task.result !== null)
  if (results.length === 0) return ''
  const total = results.reduce((sum, task) => sum + (task.result ?? 0), 0)
  return stage === 'convert' ? `${total} OCR pages` : `${total} chunks`
}
