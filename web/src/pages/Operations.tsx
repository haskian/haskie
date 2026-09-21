import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ACTIVE_JOB_STATUSES, MAX_PAGE_SIZE, api, type JobKind, type JobKindSummary, type JobRow, type Task } from '../api'
import type { PageProps } from '../App'
import { usePoll } from '../hooks/usePoll'
import { Picker, Shell } from '../ui'
import './Operations.css'
import { groupJobs, type GroupBy } from './operations/group'
import { Job } from './operations/Job'
import { errorText } from '../format'

const PAGE_SIZE = 20 // one page per kind, as the backend lists them
// The kinds whose jobs belong to one collection; the others ignore the filter (a model download
// and an archive round are no collection's work), so their jobs stay listed while it is set.
const BY_COLLECTION: ReadonlySet<JobKind> = new Set<JobKind>(['document', 'collection', 'maintenance'])

const GROUPS: { id: GroupBy; label: string }[] = [
  { id: 'status', label: 'Status' },
  { id: 'kind', label: 'Operation' },
  { id: 'day', label: 'Day' },
]

/** Every batch of an operation: its stage jobs' tasks, one read per distinct workflow. */
const operationTasks = (job: JobRow): Promise<Task[]> =>
  Promise.all([...new Set(job.stages.map((stage) => stage.job_id))].map((id) => api.jobTasks(id))).then((lists) => lists.flat())

/**
 * Every kind of background work in one list: merged, sorted newest first, and grouped by status,
 * kind or day. While anything is running the whole listing is re-read every two seconds.
 */
export function Operations({ route, counts }: PageProps) {
  const [kinds, setKinds] = useState<JobKindSummary[]>([])
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [tasks, setTasks] = useState<Record<string, Task[]>>({})
  const [hasMore, setHasMore] = useState(false)
  const [pages, setPages] = useState(1)
  const [collections, setCollections] = useState<string[]>([])
  const [collection, setCollection] = useState('')
  const [groupBy, setGroupBy] = useState<GroupBy>('status')
  const [error, setError] = useState<string | null>(null)
  const stale = useRef(new Set<string>()) // jobs whose batches were last read while they still ran

  // A batch listing that fails leaves the job without one rather than failing the whole page.
  const loadTasks = useCallback(async (wanted: JobRow[]): Promise<void> => {
    if (wanted.length === 0) return
    const loaded = await Promise.all(
      wanted.map((job) =>
        operationTasks(job)
          .then((rows) => [job.id, rows] as const)
          .catch(() => [job.id, []] as const),
      ),
    )
    for (const job of wanted) {
      if (ACTIVE_JOB_STATUSES.has(job.status)) stale.current.add(job.id)
      else stale.current.delete(job.id)
    }
    setTasks((current) => ({ ...current, ...Object.fromEntries(loaded) }))
  }, [])

  // ponytail: no cursor bookkeeping. The backend's cursor is an offset, so asking for a wider
  // first page is the same request as paging into it, and one window size serves every kind.
  const load = useCallback((): Promise<void> => {
    const pageSize = Math.min(MAX_PAGE_SIZE, PAGE_SIZE * pages)
    const filter = collection === '' ? undefined : collection
    return api
      .jobKinds()
      .then((summaries) =>
        Promise.all(
          summaries.map((summary) =>
            api.jobsByKind(summary.kind, { page_size: pageSize, collection: BY_COLLECTION.has(summary.kind) ? filter : undefined }),
          ),
        ).then((listings) => {
          const merged = listings.flatMap((listing) => listing.items).sort((a, b) => b.created_at - a.created_at)
          setKinds(summaries)
          setJobs(merged)
          setHasMore(listings.some((listing) => listing.next_cursor !== null))
          setError(null)
          return loadTasks(
            merged.filter((job) => job.kind === 'document' && (ACTIVE_JOB_STATUSES.has(job.status) || stale.current.has(job.id))),
          )
        }),
      )
      .catch((failure: unknown) => setError(errorText(failure)))
  }, [pages, collection, loadTasks])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    api.collectionNames().then(setCollections).catch(() => undefined)
  }, [])

  // The backend's count sees jobs no page holds yet; the rows on screen keep the poll going while
  // one of them finishes.
  const running = kinds.reduce((n, summary) => n + summary.active, 0) + jobs.filter((job) => ACTIVE_JOB_STATUSES.has(job.status)).length
  usePoll(running > 0, load)

  // An active job's batches are read by every poll. A finished one's are read by the click that
  // unfolds it, and once more by the poll that sees it finish, or the last in-flight list would
  // stay on screen.
  const toggle = useCallback((job: JobRow, open: boolean): void => {
    if (open && job.kind === 'document') void loadTasks([job])
  }, [loadTasks])

  const cancel = useCallback(
    (job: JobRow): void => {
      api
        .deleteJob(job.id)
        .then(load)
        .catch((failure: unknown) => setError(errorText(failure)))
    },
    [load],
  )

  const groups = useMemo(() => groupJobs(jobs, groupBy, kinds), [jobs, groupBy, kinds])

  const side = (
    <>
      <section>
        <span className="label label-mono">Group by</span>
        <nav className="nav" id="group-by">
          {GROUPS.map((group) => (
            <button
              key={group.id}
              className="nav-item"
              type="button"
              data-group={group.id}
              aria-current={group.id === groupBy ? 'true' : undefined}
              onClick={() => setGroupBy(group.id)}
            >
              {group.label}
            </button>
          ))}
        </nav>
      </section>
      <section>
        <span className="label label-mono">Collection</span>
        <Picker
          ariaLabel="Collection"
          options={[{ value: '', label: 'All collections' }, ...collections.map((name) => ({ value: name, label: name }))]}
          value={collection}
          onChange={setCollection}
        />
      </section>
    </>
  )

  return (
    <Shell current={route.name} counts={counts} side={side}>
      <div className="operations sections">
        {error !== null && <p className="muted">{error}</p>}
        {error === null && groups.length === 0 && <p className="muted">No jobs yet</p>}
        {groups.map((group) => (
          <section className="operation-section section" key={group.key} data-status={group.key}>
            <span className="mono muted">{group.key}</span>
            {group.jobs.map((job) => (
              <Job key={job.id} job={job} tasks={tasks[job.id] ?? null} onToggle={toggle} onCancel={cancel} />
            ))}
          </section>
        ))}
        {hasMore && (
          <div className="row">
            <button className="btn btn-ghost" type="button" onClick={() => setPages((page) => page + 1)}>
              Load more
            </button>
          </div>
        )}
      </div>
    </Shell>
  )
}
