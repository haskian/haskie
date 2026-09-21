import type { JobKindSummary, JobRow } from '../../api'
import { day } from '../../format'

export type GroupBy = 'status' | 'kind' | 'day'

// DBOS statuses, in the words the page groups them under. Anything else keeps its own name.
const STATUS_LABELS: Record<string, string> = {
  ENQUEUED: 'Running',
  PENDING: 'Running',
  SUCCESS: 'Completed',
  ERROR: 'Failed',
  CANCELLED: 'Cancelled',
}
const STATUS_ORDER: readonly string[] = ['Running', 'Completed', 'Failed', 'Cancelled']

export const statusGroup = (job: JobRow): string => STATUS_LABELS[job.status] ?? job.status

/** The day a job started, without the time: "Sat 19 Jan". */
export const dayGroup = (job: JobRow): string => day(job.created_at)

export interface JobGroup {
  key: string
  jobs: JobRow[]
}

/**
 * The sections of the listing. Jobs arrive newest first, which is the order within every section
 * and the order of the day sections; status and kind have an order of their own.
 */
export function groupJobs(jobs: JobRow[], by: GroupBy, kinds: JobKindSummary[]): JobGroup[] {
  if (by === 'kind') {
    return kinds
      .map((summary) => ({ key: summary.label, jobs: jobs.filter((job) => job.kind === summary.kind) }))
      .filter((group) => group.jobs.length > 0)
  }
  const keyOf = by === 'status' ? statusGroup : dayGroup
  const groups = new Map<string, JobRow[]>()
  for (const job of jobs) {
    const key = keyOf(job)
    const bucket = groups.get(key)
    if (bucket) bucket.push(job)
    else groups.set(key, [job])
  }
  const sections = [...groups].map(([key, rows]) => ({ key, jobs: rows }))
  if (by === 'day') return sections
  const rank = (key: string): number => (STATUS_ORDER.includes(key) ? STATUS_ORDER.indexOf(key) : STATUS_ORDER.length)
  return sections.sort((a, b) => rank(a.key) - rank(b.key))
}
