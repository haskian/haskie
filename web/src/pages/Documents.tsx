import { useCallback, useState } from 'react'
import {
  ACTIVE_DOCUMENT_STATUSES,
  api,
  DOCUMENT_STATUSES,
  type DocStatus,
  type Document,
  type EmbeddingEntry,
} from '../api'
import type { Route } from '../App'
import { Pager, SortHeader } from '../components/Pager'
import { bytes } from '../format'
import { useBulkJob } from '../hooks/useBulkJob'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'
import { useRun } from '../hooks/useRun'

// An import can be re-run only from a state it stopped in; every other status is the pipeline's.
const RETRYABLE: readonly DocStatus[] = ['error', 'cancelled']

// What one document's expanded row shows: the collections holding it, and the embeddings it has
// on disk. Both are one request each, so a row asks for them only when it is opened.
interface Details {
  collections: string[]
  embeddings: EmbeddingEntry[]
}

// Every document in the home, imported once and shared by the collections that hold it.
export function Documents({ navigate }: { navigate: (p: Route) => void }) {
  const [status, setStatus] = useState<DocStatus | ''>('')
  const [renameTo, setRenameTo] = useState('')
  const [description, setDescription] = useState('')
  const [path, setPath] = useState('')
  const [open, setOpen] = useState<string | null>(null)
  const [details, setDetails] = useState<Details | null>(null)

  const docs = usePaged((q) => api.documents({ ...q, status: status || undefined }), { sort: 'name', deps: [status] })
  const refresh = docs.refresh
  const { run, busy, error, setError } = useRun(refresh)

  // Anything still in the pipeline keeps the listing fresh; so does a deletion until it is gone.
  const running = docs.items.some((d) => ACTIVE_DOCUMENT_STATUSES.includes(d.status) || d.status === 'deleting')
  usePoll(running, refresh)

  // A deletion is accepted (202) and runs in the background, so the page follows the job; the
  // listing is re-read once it is really gone.
  const onDeleted = useCallback(() => {
    void refresh()
  }, [refresh])
  const deletion = useBulkJob(onDeleted, setError)

  // Upload is two steps: the bytes are staged first, then named and imported. One name would be
  // taken repeatedly by a batch, so only a single file may be renamed.
  const upload = (files: File[]) =>
    run(async () => {
      const rename = files.length === 1 ? renameTo.trim() : ''
      await Promise.all(
        files.map(async (file) => {
          const staged = await api.stageUpload(file)
          await api.importStaged({
            staging_id: staged.staging_id,
            name: rename || undefined,
            description: description.trim() || undefined,
          })
        }),
      )
      setRenameTo('')
      setDescription('')
    })

  const toggle = (doc: string) => {
    if (open === doc) {
      setOpen(null)
      setDetails(null)
      return
    }
    setOpen(doc)
    setDetails(null)
    Promise.all([api.documentCollections(doc), api.documentEmbeddings(doc)])
      .then(([collections, embeddings]) => setDetails({ collections, embeddings }))
      .catch((e) => setError(String(e)))
  }

  const remove = (doc: string) => run(() => deletion.start(() => api.deleteDocument(doc)))

  return (
    <div>
      <h2>Documents</h2>
      <p className="muted">
        A document is imported once. Attach it to as many collections as you like; they share its
        conversion, and its embeddings whenever they chunk it the same way.
      </p>
      {error && <p className="error">{error}</p>}

      <h3>Import</h3>
      <p>
        <input
          type="file"
          multiple
          disabled={busy}
          onChange={(e) => {
            const files = Array.from(e.target.files ?? [])
            if (files.length > 0) upload(files)
            e.target.value = ''
          }}
        />{' '}
        <input placeholder="rename to (one file)" value={renameTo} onChange={(e) => setRenameTo(e.target.value)} />{' '}
        <input placeholder="description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />
      </p>
      <p>
        <input
          className="path-input"
          placeholder="or import a path the server can read"
          value={path}
          onChange={(e) => setPath(e.target.value)}
        />{' '}
        <button
          disabled={busy || !path.trim()}
          onClick={() =>
            run(async () => {
              await api.importPath(path.trim(), { description: description.trim() || undefined })
              setPath('')
              setDescription('')
            })
          }
        >
          Import path
        </button>
      </p>

      <h3>
        All documents {deletion.running && <span className="muted">· deleting…</span>}
      </h3>
      <p>
        <select value={status} onChange={(e) => setStatus(e.target.value as DocStatus | '')}>
          <option value="">all statuses</option>
          {DOCUMENT_STATUSES.map((s) => (
            <option key={s} value={s}>
              {s}
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
            <DocumentRow
              key={d.name}
              doc={d}
              details={open === d.name ? details : null}
              expanded={open === d.name}
              onToggle={() => toggle(d.name)}
              onOpen={() => navigate({ name: 'viewer', doc: d.name })}
              onDescribe={(text) => run(() => api.describeDocument(d.name, text))}
              onRetry={() => run(() => api.reimportDocument(d.name))}
              onDelete={() => {
                if (confirm(`Delete "${d.name}" and remove it from every collection?`)) remove(d.name)
              }}
            />
          ))}
        </tbody>
      </table>
      <Pager paged={docs} />
    </div>
  )
}

function DocumentRow({
  doc: d,
  details,
  expanded,
  onToggle,
  onOpen,
  onDescribe,
  onRetry,
  onDelete,
}: {
  doc: Document
  details: Details | null
  expanded: boolean
  onToggle: () => void
  onOpen: () => void
  onDescribe: (text: string) => void
  onRetry: () => void
  onDelete: () => void
}) {
  return (
    <>
      <tr>
        <td>
          <button className="link" onClick={onOpen}>
            {d.name}
          </button>
        </td>
        <td>
          <input
            className="description-edit"
            placeholder="—"
            defaultValue={d.description}
            onBlur={(e) => {
              if (e.target.value !== d.description) onDescribe(e.target.value)
            }}
          />
        </td>
        <td className="muted">{bytes.format(d.size)}</td>
        <td className={d.status === 'error' ? 'error' : 'muted'}>
          {d.status}
          {d.error && <pre className="error-detail">{d.error}</pre>}
        </td>
        <td className="muted">{d.updated_at ? new Date(d.updated_at * 1000).toLocaleString() : ''}</td>
        <td>
          <button onClick={onToggle}>{expanded ? 'hide' : 'collections'}</button>
          {RETRYABLE.includes(d.status) && <button onClick={onRetry}>retry import</button>}
          <button disabled={d.status === 'deleting'} onClick={onDelete}>
            delete
          </button>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td></td>
          <td colSpan={5}>
            {details === null ? (
              <span className="muted">loading…</span>
            ) : (
              <>
                <div className="tags">
                  {details.collections.length === 0 ? (
                    <span className="muted">in no collection yet</span>
                  ) : (
                    details.collections.map((c) => <span key={c} className="tag">{c}</span>)
                  )}
                </div>
                <div className="hit-meta muted">
                  {details.embeddings.length === 0 ? (
                    <span>no embeddings cached</span>
                  ) : (
                    details.embeddings.map((e) => (
                      <span key={e.id} title={e.urn}>
                        {e.model} · {e.chunker} {e.chunk_size}/{e.chunk_overlap} · {e.rows} rows
                      </span>
                    ))
                  )}
                </div>
              </>
            )}
          </td>
        </tr>
      )}
    </>
  )
}
