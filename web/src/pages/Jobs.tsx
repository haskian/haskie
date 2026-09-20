import { useCallback, useEffect, useState } from 'react'
import { api, type JobKind, type JobKindSummary, type JobRow, type Task } from '../api'
import { JobLine } from '../components/JobLine'
import { Pager } from '../components/Pager'
import { useOptions } from '../hooks/useOptions'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'

const PAGE_SIZE = 20 // one section per kind: each shows a first page and loads more on demand

// The kinds whose jobs belong to one collection; the others ignore the filter (a model download is
// no collection's work), so their section keeps its rows while it is set.
const BY_COLLECTION: ReadonlySet<JobKind> = new Set<JobKind>(['document', 'collection', 'maintenance'])

// One section per kind of background work, in the order the backend lists them. Each section
// pages and polls on its own; this view only keeps the counts, which is what tells it that work
// no section has on screen yet has started.
export function Jobs() {
  const [kinds, setKinds] = useState<JobKindSummary[]>([])
  const [collections, setCollections] = useState<string[]>([])
  const [collection, setCollection] = useState('')

  const reloadKinds = useCallback(() => api.jobKinds().then(setKinds).then(() => undefined), [])
  useEffect(() => {
    void reloadKinds()
    api.collectionNames().then(setCollections)
  }, [reloadKinds])

  const active = kinds.reduce((n, k) => n + k.active, 0)
  usePoll(active > 0, reloadKinds)

  return (
    <div>
      <h2>Jobs</h2>
      <p>
        <select value={collection} onChange={(e) => setCollection(e.target.value)}>
          <option value="">all collections</option>
          {collections.map((c) => <option key={c}>{c}</option>)}
        </select>{' '}
        <span className="muted">{active} active</span>
      </p>
      {kinds.map((summary) => (
        <JobSection
          key={summary.kind}
          summary={summary}
          collection={BY_COLLECTION.has(summary.kind) ? collection : ''}
        />
      ))}
    </div>
  )
}

function JobSection({ summary, collection }: { summary: JobKindSummary; collection: string }) {
  const { kind, label } = summary
  const { active_job_statuses } = useOptions()
  const [open, setOpen] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])

  const jobs = usePaged<JobRow>((q) => api.jobsByKind(kind, { ...q, collection: collection || undefined }), { pageSize: PAGE_SIZE, deps: [kind, collection] })
  const reload = jobs.refresh
  const running = jobs.items.filter((j) => active_job_statuses.includes(j.status)).length

  // one refresh for both: the rows already on screen, and the batches of the open job
  const refresh = useCallback(
    () => Promise.all([reload(), open ? api.jobTasks(open).then(setTasks) : Promise.resolve()]).then(() => undefined),
    [reload, open],
  )
  // the backend's count sees jobs this section has not listed yet; the rows on screen keep the
  // poll going while one of them finishes
  usePoll(running > 0 || summary.active > 0, refresh)

  // the batches of one job are loaded by the click that opened it, not by an effect
  const openJob = useCallback((id: string | null) => {
    setOpen(id)
    setTasks([])
    if (id) api.jobTasks(id).then(setTasks)
  }, [])

  if (jobs.items.length === 0) {
    return <h3 className="muted">{label} — none</h3>
  }
  return (
    <section>
      <h3>
        {label} {summary.active > 0 && <span className="muted">({summary.active} active)</span>}
      </h3>
      {jobs.error && <p className="error">{jobs.error}</p>}
      <table className="jobs">
        <thead>
          <tr>
            <th>started</th>
            <th>job</th>
            <th>status</th>
            <th>progress</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {jobs.items.map((j) => (
            <JobLine key={j.id} job={j} tasks={open === j.id ? tasks : null} onToggle={() => openJob(open === j.id ? null : j.id)} onCancel={() => api.deleteJob(j.id).then(refresh)} />
          ))}
        </tbody>
      </table>
      <Pager paged={jobs} />
    </section>
  )
}
