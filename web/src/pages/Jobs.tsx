import { useCallback, useEffect, useRef, useState } from 'react'
import { ACTIVE_JOB_STATUSES as ACTIVE, api, type JobKind, type JobKindSummary, type JobRow, type Task } from '../api'
import { JobLine } from '../components/JobLine'
import { Pager } from '../components/Pager'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'

const PAGE_SIZE = 20 // one section per kind: each shows a first page and loads more on demand

// The kinds whose jobs belong to one collection; the others ignore the filter (a model download
// and an archive round are no collection's work), so their section keeps its rows while it is set.
const BY_COLLECTION: ReadonlySet<JobKind> = new Set<JobKind>(['document', 'collection', 'maintenance'])

// One section per kind of background work, in the order the backend lists them. Each section
// pages on its own, and one poll re-reads every one of them while anything is running: a document
// pipeline, a whole-collection job, a model download, a maintenance run or an archive round.
export function Jobs() {
  const [kinds, setKinds] = useState<JobKindSummary[]>([])
  const [collections, setCollections] = useState<string[]>([])
  const [collection, setCollection] = useState('')
  const [onScreen, setOnScreen] = useState<Record<string, number>>({})
  const refreshers = useRef(new Map<JobKind, () => Promise<void>>())

  // a section reports its refresh and how much of it is running; the guard keeps an unchanged
  // count from starting another render
  const register = useCallback((kind: JobKind, refresh: () => Promise<void>, running: number) => {
    refreshers.current.set(kind, refresh)
    setOnScreen((counts) => (counts[kind] === running ? counts : { ...counts, [kind]: running }))
  }, [])

  const refreshAll = useCallback(async () => {
    await Promise.all([...refreshers.current.values()].map((refresh) => refresh()))
    setKinds(await api.jobKinds())
  }, [])

  const reloadKinds = useCallback(() => api.jobKinds().then(setKinds), [])
  useEffect(() => {
    void reloadKinds()
    api.collectionNames().then(setCollections)
  }, [reloadKinds])

  // the backend's count sees jobs no section has on screen yet; the rows on screen keep the poll
  // going while one of them finishes
  const active = kinds.reduce((n, k) => n + k.active, 0) + Object.values(onScreen).reduce((n, c) => n + c, 0)
  usePoll(active > 0, refreshAll)

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
          onState={register}
        />
      ))}
    </div>
  )
}


function JobSection({ summary, collection, onState }: { summary: JobKindSummary; collection: string; onState: (kind: JobKind, refresh: () => Promise<void>, running: number) => void }) {
  const { kind, label } = summary
  const [open, setOpen] = useState<string | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])

  const jobs = usePaged<JobRow>((q) => api.jobsByKind(kind, { ...q, collection: collection || undefined }), { pageSize: PAGE_SIZE, deps: [kind, collection] })
  const reload = jobs.refresh
  const running = jobs.items.filter((j) => ACTIVE.has(j.status)).length

  // one refresh for both: the rows already on screen, and the batches of the open job
  const refresh = useCallback(
    () => Promise.all([reload(), open ? api.jobTasks(open).then(setTasks) : Promise.resolve()]).then(() => undefined),
    [reload, open],
  )
  useEffect(() => {
    onState(kind, refresh, running)
  }, [kind, onState, refresh, running])

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
