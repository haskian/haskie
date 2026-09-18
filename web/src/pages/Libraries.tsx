import { useCallback, useEffect, useState } from 'react'
import { ACTIVE_DOCUMENT_STATUSES, ACTIVE_JOB_STATUSES, api, type BulkJob, type BulkStarted, DOCUMENT_STATUSES, type DocStatus, type JobRow, type LibraryInfo, type LibrarySettings, type Options } from '../api'
import type { Route } from '../App'
import { ConversionSettingsForm } from '../components/ConversionSettingsForm'
import { Pager, SortHeader } from '../components/Pager'
import { Search } from '../components/Search'
import { SearchSettingsForm } from '../components/SearchSettingsForm'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'
import { JobLine } from '../components/JobLine'

const bytes = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  style: 'unit',
  unit: 'byte',
  unitDisplay: 'narrow',
})
const JOBS_PAGE_SIZE = 20 // the newest jobs of one library, as a hint; the Jobs view pages them all

export function Libraries({ initial, navigate }: { initial?: string; navigate: (p: Route) => void }) {
  const [selected, setSelected] = useState<string | undefined>(initial)
  const [newName, setNewName] = useState('')
  const [newDescription, setNewDescription] = useState('')
  const libraries = usePaged((q) => api.libraries(q), { sort: 'name' })
  const refresh = libraries.refresh

  const create = async () => {
    const info = await api.createLibrary(newName, newDescription)
    setNewName('')
    setNewDescription('')
    await refresh()
    setSelected(info.name)
  }

  return (
    <div className="split">
      <aside>
        <h2>Libraries</h2>
        {libraries.error && <p className="error">{libraries.error}</p>}
        <ul className="list">
          {libraries.items.map((l) => (
            <li key={l.name}>
              <button className={l.name === selected ? 'active' : ''} onClick={() => setSelected(l.name)}>
                {l.name} <span className="muted">{l.counts.total}</span>
                {l.counts.active > 0 && <span className="muted"> · {l.counts.active} active</span>}
                {l.description && <small className="muted description">{l.description}</small>}
              </button>
            </li>
          ))}
        </ul>
        <Pager paged={libraries} />
        <form
          onSubmit={(e) => {
            e.preventDefault()
            create()
          }}
        >
          <input placeholder="new library" value={newName} onChange={(e) => setNewName(e.target.value)} />
          <input placeholder="what it holds (optional)" value={newDescription} onChange={(e) => setNewDescription(e.target.value)} />
          <button disabled={!newName.trim()}>Create</button>
        </form>
      </aside>
      <section>
        {selected ? (
          <LibraryDetail
            key={selected}
            name={selected}
            navigate={navigate}
            onDeleted={() => {
              setSelected(undefined)
              refresh()
            }}
          />
        ) : (
          <p className="muted">Select or create a library.</p>
        )}
      </section>
    </div>
  )
}

function LibraryDetail({
  name,
  navigate,
  onDeleted,
}: {
  name: string
  navigate: (p: Route) => void
  onDeleted: () => void
}) {
  const [info, setInfo] = useState<LibraryInfo | null>(null)
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<DocStatus | ''>('')
  const [renameTo, setRenameTo] = useState('')
  const [description, setDescription] = useState('')
  // "Index all" and "Delete library" are accepted (202) and run in the background, so the page
  // follows the job instead of waiting for the request.
  const [bulk, setBulk] = useState<BulkJob | null>(null)
  const bulkRunning = bulk !== null && ACTIVE_JOB_STATUSES.has(bulk.status)

  // counts and settings for the header, jobs for the block below; the documents are paged
  const refreshInfo = useCallback(
    () =>
      Promise.all([api.library(name), api.jobsByKind('document', { library: name, page_size: JOBS_PAGE_SIZE })]).then(([i, j]) => {
        setInfo(i)
        setJobs(j.items)
      }),
    [name],
  )
  const docs = usePaged((q) => api.documents(name, { ...q, status: status || undefined }), {
    sort: 'name',
    deps: [name, status],
  })
  const refreshDocs = docs.refresh
  const refresh = useCallback(() => Promise.all([refreshInfo(), refreshDocs()]), [refreshInfo, refreshDocs])

  useEffect(() => {
    refreshInfo()
    api.options().then(setOptions)
  }, [refreshInfo])

  // poll while anything is in the pipeline; the counts come from the database, not from the page
  usePoll((info?.counts.active ?? 0) > 0, refresh)

  // follow a bulk job to its end: an index hands over to the document poll above, a deletion
  // takes the library off the page once it is really gone
  const followBulk = useCallback(() => {
    if (!bulk) return
    api
      .jobProgress(bulk.id)
      .then((next) => {
        setBulk(next)
        if (ACTIVE_JOB_STATUSES.has(next.status)) return
        if (next.kind === 'delete_library') onDeleted()
        else refresh()
      })
      .catch((e) => setError(String(e)))
  }, [bulk, onDeleted, refresh])
  usePoll(bulkRunning, followBulk)

  const run = async (fn: () => Promise<unknown>) => {
    setError(null)
    try {
      await fn()
      await refresh()
    } catch (e) {
      setError(String(e))
    }
  }

  const startBulk = async (start: () => Promise<BulkStarted>) => {
    setError(null)
    try {
      const { job_id } = await start()
      setBulk(await api.jobProgress(job_id))
    } catch (e) {
      setError(String(e))
    }
  }

  if (!info) return null

  return (
    <>
      <h2>{info.name}</h2>
      <p>
        <input
          className="description-edit"
          placeholder="what this library holds"
          defaultValue={info.description}
          onBlur={(e) => {
            const next = e.target.value
            if (next !== info.description) run(() => api.describeLibrary(name, next))
          }}
        />
      </p>
      {error && <p className="error">{error}</p>}
      {info.index_outdated && (
        <p className="banner">index was built by an older version — use "Index all" to rebuild it</p>
      )}

      <h3>Search</h3>
      <Search run={(q) => api.searchLibrary(name, q)} libraries={[name]} placeholder={`search ${name} (top ${info.search.limit})`} />

      <h3>
        Documents <span className="muted">{info.counts.total} total · {info.counts.indexed} indexed · {info.counts.active} active · {info.counts.error} failed</span>
      </h3>
      <p>
        <input
          type="file"
          multiple
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            // one file may be renamed; renaming a batch to one name would overwrite it repeatedly
            const rename = files.length === 1 ? renameTo.trim() : ''
            run(async () => {
              await Promise.all(
                files.map((f) => api.upload(name, f, { rename_to: rename || undefined, description: description.trim() || undefined })),
              )
              setRenameTo('')
              setDescription('')
            })
            e.target.value = ''
          }}
        />{' '}
        <input placeholder="rename to (one file)" value={renameTo} onChange={(e) => setRenameTo(e.target.value)} />{' '}
        <input placeholder="description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />{' '}
        <select value={status} onChange={(e) => setStatus(e.target.value as DocStatus | '')}>
          <option value="">all statuses</option>
          {DOCUMENT_STATUSES.map((s) => (
            <option key={s} value={s}>
              {s} ({info.counts.by_status[s] ?? 0})
            </option>
          ))}
        </select>
      </p>
      {docs.error && <p className="error">{docs.error}</p>}
      <table>
        <thead>
          <tr>
            <SortHeader field="name" label="name" paged={docs} />
            <th>description</th>
            <SortHeader field="size" label="size" paged={docs} />
            <SortHeader field="status" label="status" paged={docs} />
            <SortHeader field="updated_at" label="updated" paged={docs} />
            <th></th>
          </tr>
        </thead>
        <tbody>
          {docs.items.map((d) => (
            <tr key={d.name}>
              <td>
                <button className="link" onClick={() => navigate({ name: 'viewer', library: name, doc: d.name })}>
                  {d.name}
                </button>
              </td>
              <td>
                <input
                  className="description-edit"
                  placeholder="—"
                  defaultValue={d.description}
                  onBlur={(e) => {
                    const next = e.target.value
                    if (next !== d.description) run(() => api.describeDocument(name, d.name, next))
                  }}
                />
              </td>
              <td className="muted">{bytes.format(d.size)}</td>
              <td className={d.status === 'error' ? 'error' : 'muted'}>
                {d.status === 'uploaded' ? 'preview' : d.status}
                {d.error && <pre className="error-detail">{d.error}</pre>}
              </td>
              <td className="muted">{d.updated_at ? new Date(d.updated_at * 1000).toLocaleString() : ''}</td>
              <td>
                <button
                  disabled={ACTIVE_DOCUMENT_STATUSES.includes(d.status)}
                  onClick={() => run(() => api.indexDocument(name, d.name))}
                >
                  {d.status === 'indexed' ? 'reindex' : 'index'}
                </button>
                <button onClick={() => run(() => api.deleteDocument(name, d.name))}>delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Pager paged={docs} />

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
        <LibrarySettingsForm
          settings={info.settings}
          effective={info.effective}
          searchDefault={info.search}
          options={options}
          onSave={(s) => run(() => api.saveLibrarySettings(name, s))}
        />
      )}
      <p>
        <button disabled={bulkRunning} onClick={() => startBulk(() => api.indexLibrary(name))}>
          Index all
        </button>{' '}
        <button
          disabled={bulkRunning}
          onClick={() => {
            if (confirm(`Delete library "${name}" and all its files?`)) startBulk(() => api.deleteLibrary(name))
          }}
        >
          Delete library
        </button>{' '}
        {bulk && <BulkStatus job={bulk} />}
      </p>
    </>
  )
}

function BulkStatus({ job }: { job: BulkJob }) {
  const running = ACTIVE_JOB_STATUSES.has(job.status)
  const what = job.kind === 'delete_library' ? 'deleting' : 'queueing documents'
  return (
    <span className={job.error ? 'error' : 'muted'}>
      {running ? `${what}…` : `${what}: ${job.status.toLowerCase()}`}
      {job.progress && ` ${job.progress.done}/${job.progress.total}`}
      {job.error && ` — ${job.error}`}
    </span>
  )
}

function LibrarySettingsForm({
  settings,
  effective,
  searchDefault,
  options,
  onSave,
}: {
  settings: LibrarySettings
  effective: LibraryInfo['effective']
  searchDefault: LibraryInfo['search']
  options: Options
  onSave: (s: LibrarySettings) => void
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
      <ConversionSettingsForm
        value={draft}
        defaults={effective}
        options={options}
        onChange={(next) => setDraft({ ...draft, ...(next as Partial<LibrarySettings>) })}
      />
      <h4>Search</h4>
      <SearchSettingsForm
        value={draft.search}
        defaults={searchDefault}
        options={options}
        onChange={(next) => setDraft({ ...draft, search: next as LibrarySettings['search'] })}
      />
      <button>Save</button>
    </form>
  )
}
