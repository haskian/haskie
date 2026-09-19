import { useCallback, useEffect, useState } from 'react'
import {
  ACTIVE_JOB_STATUSES,
  ACTIVE_MEMBER_STATUSES,
  api,
  type BulkJob,
  type BulkStarted,
  type ChunkSettings,
  type CollectionInfo,
  type CollectionSettings,
  type Document,
  type JobRow,
  MAX_PAGE_SIZE,
  MEMBER_STATUSES,
  type MemberStatus,
  type Options,
} from '../api'
import type { Route } from '../App'
import { ChunkSettingsForm } from '../components/ChunkSettingsForm'
import { JobLine } from '../components/JobLine'
import { Pager, SortHeader } from '../components/Pager'
import { Search } from '../components/Search'
import { SearchSettingsForm } from '../components/SearchSettingsForm'
import { bytes } from '../format'
import { useBulkJob } from '../hooks/useBulkJob'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'
import { useRun } from '../hooks/useRun'

const JOBS_PAGE_SIZE = 20 // the newest jobs of one collection, as a hint; the Jobs view pages them all

export function Collections({ initial, navigate }: { initial?: string; navigate: (p: Route) => void }) {
  const [selected, setSelected] = useState<string | undefined>(initial)
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const collections = usePaged((q) => api.collections(q), { sort: 'name' })
  const refresh = collections.refresh

  const create = async () => {
    const info = await api.createCollection(newName, newDescription)
    setNewName('')
    setNewDescription('')
    await refresh()
    setSelected(info.name)
  }

  // the detail view follows its bulk jobs by polling: its callbacks must keep one identity, or
  // every tick tears the interval down and builds it again
  const onDeleted = useCallback(() => {
    setSelected(undefined)
    void refresh()
  }, [refresh])

  return (
    <div className="split">
      <aside>
        <h2>Collections</h2>
        {collections.error && <p className="error">{collections.error}</p>}
        <ul className="list">
          {collections.items.map((c) => (
            <li key={c.name}>
              <button className={c.name === selected ? 'active' : ''} onClick={() => setSelected(c.name)}>
                {c.name} <span className="muted">{c.counts.total}</span>
                {c.counts.active > 0 && <span className="muted"> · {c.counts.active} active</span>}
                {c.description && <small className="muted description">{c.description}</small>}
              </button>
            </li>
          ))}
        </ul>
        <Pager paged={collections} />
        <form
          onSubmit={(e) => {
            e.preventDefault()
            create()
          }}
        >
          <input placeholder="new collection" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <input placeholder="what it holds (optional)" value={newDescription} onChange={(e) => setNewDescription(e.target.value)} />
          <button disabled={!newName.trim()}>Create</button>
        </form>
      </aside>
      <section>
        {selected ? (
          <CollectionDetail
            key={selected}
            name={selected}
            navigate={navigate}
            onDeleted={onDeleted}
          />
        ) : (
          <p className="muted">Select or create a collection.</p>
        )}
      </section>
    </div>
  )
}

function CollectionDetail({
  name,
  navigate,
  onDeleted,
}: {
  name: string
  navigate: (p: Route) => void
  onDeleted: () => void
}) {
  const [info, setInfo] = useState<CollectionInfo | null>(null)
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [status, setStatus] = useState<MemberStatus | ''>('')
  // imported documents this collection does not hold yet, and the one picked in the dropdown
  const [candidates, setCandidates] = useState<Document[]>([])
  const [toAttach, setToAttach] = useState('')

  // counts and settings for the header, jobs for the block below; the members are paged
  const refreshInfo = useCallback(
    () =>
      Promise.all([api.collection(name), api.jobsByKind('document', { collection: name, page_size: JOBS_PAGE_SIZE })]).then(([i, j]) => {
        setInfo(i)
        setJobs(j.items)
      }),
    [name],
  )
  const members = usePaged((q) => api.collectionDocuments(name, { ...q, status: status || undefined }), {
    sort: 'name',
    deps: [name, status],
  })
  const refreshMembers = members.refresh

  // The picker needs every member name, not the page on screen, so both sides are asked for in
  // one large page each; a document may be attached to a collection only once it is imported.
  // Two full listings are too expensive to poll, so this runs on mount and after a mutation only.
  // ponytail: the ceiling is the client-side diff of two capped listings — once a collection (or
  // the home) outgrows MAX_PAGE_SIZE, the server has to answer it with a `not_in` filter instead.
  const refreshCandidates = useCallback(
    () =>
      Promise.all([
        api.documents({ status: 'imported', page_size: MAX_PAGE_SIZE, sort: 'name' }),
        api.collectionDocuments(name, { page_size: MAX_PAGE_SIZE, sort: 'name' }),
      ]).then(([imported, held]) => {
        const names = new Set(held.items.map((m) => m.document.name))
        setCandidates(imported.items.filter((d) => !names.has(d.name)))
      }),
    [name],
  )

  // what a poll re-reads: only what background work moves
  const refresh = useCallback(() => Promise.all([refreshInfo(), refreshMembers()]), [refreshInfo, refreshMembers])
  // what a mutation re-reads: the same, plus the picker the mutation may have changed
  const refreshAll = useCallback(() => Promise.all([refresh(), refreshCandidates()]), [refresh, refreshCandidates])
  const { run, error, setError } = useRun(refreshAll)

  useEffect(() => {
    refreshInfo()
    refreshCandidates()
    api.options().then(setOptions)
  }, [refreshInfo, refreshCandidates])

  // poll while anything is being written into this collection; the counts come from the database
  usePoll((info?.counts.active ?? 0) > 0, refresh)

  // "Index all" and "Delete collection" are accepted (202) and run in the background, so the page
  // follows the job instead of waiting for the request. An index hands over to the member poll
  // above; a deletion takes the collection off the page once it is really gone.
  const onBulkDone = useCallback(
    (job: BulkJob) => {
      if (job.kind === 'delete_collection') onDeleted()
      else void refreshAll()
    },
    [onDeleted, refreshAll],
  )
  const bulk = useBulkJob(onBulkDone, setError)

  const startBulk = (start: () => Promise<BulkStarted>) => {
    setError(null)
    bulk.start(start).catch((e: unknown) => setError(String(e)))
  }

  if (!info) return null

  return (
    <>
      <h2>{info.name}</h2>
      <p>
        <input
          className="description-edit"
          placeholder="what this collection holds"
          defaultValue={info.description}
          onBlur={(e) => {
            const next = e.target.value
            if (next !== info.description) run(() => api.describeCollection(name, next))
          }}
        />
      </p>
      {error && <p className="error">{error}</p>}
      {info.index_outdated && (
        <p className="banner">index was built by an older version — use "Index all" to rebuild it</p>
      )}

      <h3>Search</h3>
      <Search run={(q) => api.searchCollection(name, q)} collections={[name]} placeholder={`search ${name} (top ${info.search.limit})`} />

      <h3>
        Documents <span className="muted">{info.counts.total} total · {info.counts.indexed} indexed · {info.counts.active} active · {info.counts.error} failed</span>
      </h3>
      <p>
        <select value={toAttach} onChange={(e) => setToAttach(e.target.value)}>
          <option value="">attach a document…</option>
          {candidates.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name}
            </option>
          ))}
        </select>{' '}
        <button
          disabled={!toAttach}
          onClick={() =>
            run(async () => {
              await api.attachDocument(name, toAttach)
              setToAttach('')
            })
          }
        >
          Attach
        </button>{' '}
        <button onClick={() => navigate({ name: 'documents' })}>Import a document…</button>{' '}
        <select value={status} onChange={(e) => setStatus(e.target.value as MemberStatus | '')}>
          <option value="">all statuses</option>
          {MEMBER_STATUSES.map((s) => (
            <option key={s} value={s}>
              {s} ({info.counts.by_status[s] ?? 0})
            </option>
          ))}
        </select>
      </p>
      {members.error && <p className="error">{members.error}</p>}
      <table>
        <thead>
          <tr>
            <SortHeader field="name" label="name" paged={members} />
            <th>description</th>
            <SortHeader field="size" label="size" paged={members} />
            <SortHeader field="status" label="in collection" paged={members} />
            <SortHeader field="updated_at" label="updated" paged={members} />
            <th></th>
          </tr>
        </thead>
        <tbody>
          {members.items.map((m) => (
            <tr key={m.document.name}>
              <td>
                <button className="link" onClick={() => navigate({ name: 'viewer', doc: m.document.name })}>
                  {m.document.name}
                </button>
              </td>
              <td className="muted">{m.document.description || '—'}</td>
              <td className="muted">{bytes.format(m.document.size)}</td>
              <td className={m.status === 'error' ? 'error' : 'muted'}>
                {m.status}
                {m.error && <pre className="error-detail">{m.error}</pre>}
              </td>
              <td className="muted">{m.updated_at ? new Date(m.updated_at * 1000).toLocaleString() : ''}</td>
              <td>
                <button
                  disabled={ACTIVE_MEMBER_STATUSES.includes(m.status)}
                  onClick={() => run(() => api.reindexMember(name, m.document.name))}
                >
                  {m.status === 'indexed' ? 'reindex' : 'index'}
                </button>
                <button onClick={() => run(() => api.detachDocument(name, m.document.name))}>detach</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Pager paged={members} />

      {jobs.length > 0 && (
        <details>
          <summary className="muted">jobs ({jobs.filter((j) => ACTIVE_JOB_STATUSES.has(j.status)).length} active)</summary>
          <table className="jobs">
            <tbody>
              {jobs.map((j) => (
                <JobLine key={j.id} job={j} tasks={null} onToggle={() => navigate({ name: 'jobs' })} onCancel={() => run(() => api.deleteJob(j.id))} />
              ))}
            </tbody>
          </table>
        </details>
      )}

      <h3>Settings</h3>
      {options && (
        <CollectionSettingsForm
          settings={info.settings}
          effective={info.effective}
          searchDefault={info.search}
          options={options}
          onSave={(s) => run(() => api.saveCollectionSettings(name, s))}
        />
      )}
      <p>
        <button disabled={bulk.running} onClick={() => startBulk(() => api.indexCollection(name))}>
          Index all
        </button>{' '}
        <button
          disabled={bulk.running}
          onClick={() => {
            if (confirm(`Delete collection "${name}"? Its documents stay; only this index goes.`)) {
              startBulk(() => api.deleteCollection(name))
            }
          }}
        >
          Delete collection
        </button>{' '}
        {bulk.job && <BulkStatus job={bulk.job} />}
      </p>
    </>
  )
}

function BulkStatus({ job }: { job: BulkJob }) {
  const running = ACTIVE_JOB_STATUSES.has(job.status)
  const what = job.kind === 'index_collection' ? 'queueing documents' : 'deleting'
  return (
    <span className={job.error ? 'error' : 'muted'}>
      {running ? `${what}…` : `${what}: ${job.status.toLowerCase()}`}
      {job.progress && ` ${job.progress.done}/${job.progress.total}`}
      {job.error && ` — ${job.error}`}
    </span>
  )
}

// Only chunking and search: conversion happened once, at import, so a collection cannot change it.
function CollectionSettingsForm({
  settings,
  effective,
  searchDefault,
  options,
  onSave,
}: {
  settings: CollectionSettings
  effective: ChunkSettings
  searchDefault: CollectionInfo['search']
  options: Options
  onSave: (s: CollectionSettings) => void
}) {
  const [draft, setDraft] = useState(settings)
  return (
    <form
      className="settings"
      onSubmit={(e) => {
        e.preventDefault()
        onSave(draft)
      }}
    >
      <h4>Chunking</h4>
      <ChunkSettingsForm nullable value={draft} defaults={effective} options={options} onChange={setDraft} />
      <h4>Search</h4>
      <SearchSettingsForm
        value={draft.search}
        defaults={searchDefault}
        options={options}
        onChange={(next) => setDraft({ ...draft, search: next as CollectionSettings['search'] })}
      />
      <button>Save</button>
    </form>
  )
}
