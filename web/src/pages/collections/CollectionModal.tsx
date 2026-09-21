import { Minus, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ACTIVE_JOB_STATUSES,
  api,
  MAX_PAGE_SIZE,
  type BulkJob,
  type BulkStarted,
  type CollectionInfo,
  type Document,
  type Member,
  type Options,
} from '../../api'
import { errorText, matchesText, needleOf } from '../../format'
import { useBulkJob } from '../../hooks/useBulkJob'
import { usePoll } from '../../hooks/usePoll'
import { useRun } from '../../hooks/useRun'
import { DescriptionBox, documentIcon, Kv, Modal, SearchBox, SearchPanel, Tabs, type TabDef } from '../../ui'
import { candidateDocuments } from './candidates'
import { SettingsForm } from './SettingsForm'

const TABS: TabDef[] = [
  { id: 'collection-documents', label: 'Documents' },
  { id: 'collection-search', label: 'Search' },
  { id: 'collection-settings', label: 'Settings' },
  { id: 'collection-info', label: 'Info' },
]

/** One collection, opened from the gallery. Mounts its content only while the route names it. */
export function CollectionModal({
  name,
  onClose,
  onChanged,
}: {
  name: string | undefined
  onClose: () => void
  onChanged: () => Promise<unknown>
}) {
  return (
    <Modal open={name !== undefined} onClose={onClose} title={name ?? ''} subtitle="collection">
      {name !== undefined && <CollectionBody key={name} name={name} onClose={onClose} onChanged={onChanged} />}
    </Modal>
  )
}

function CollectionBody({ name, onClose, onChanged }: { name: string; onClose: () => void; onChanged: () => Promise<unknown> }) {
  const [info, setInfo] = useState<CollectionInfo | null>(null)
  const [members, setMembers] = useState<Member[]>([])
  const [imported, setImported] = useState<Document[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [tab, setTab] = useState<string>(TABS[0].id)
  const [filter, setFilter] = useState('')

  // what background work moves: the counts in the header, and how far each member got
  const refreshInfo = useCallback(
    () =>
      Promise.all([api.collection(name), api.collectionDocuments(name, { page_size: MAX_PAGE_SIZE, sort: 'name' })]).then(
        ([next, page]) => {
          setInfo(next)
          setMembers(page.items)
        },
      ),
    [name],
  )
  // A document may be attached only once it is imported, so the other pane lists exactly those.
  const refreshImported = useCallback(
    () => api.documents({ status: 'imported', page_size: MAX_PAGE_SIZE, sort: 'name' }).then((page) => setImported(page.items)),
    [],
  )
  // what a mutation re-reads: this collection, plus the gallery behind the modal
  const refreshAll = useCallback(
    () => Promise.all([refreshInfo(), refreshImported(), onChanged()]),
    [refreshInfo, refreshImported, onChanged],
  )
  const { run, busy, error, setError } = useRun(refreshAll)

  // opening reads this collection alone: the gallery behind it is already on screen
  useEffect(() => {
    Promise.all([refreshInfo(), refreshImported()]).catch((cause: unknown) => setError(errorText(cause)))
    api.options().then(setOptions).catch((cause: unknown) => setError(errorText(cause)))
  }, [refreshInfo, refreshImported, setError])

  // A failed tick has no one to answer to; the next one catches up.
  const poll = useCallback(() => {
    void refreshInfo().catch(() => undefined)
  }, [refreshInfo])
  usePoll((info?.counts.active ?? 0) > 0, poll)

  // "Index all" and "Delete collection" are accepted (202) and run in the background, so the
  // modal follows the job. The callbacks keep one identity, or `useBulkJob` rebuilds its poll
  // on every tick. A deleted collection has nothing left to re-read, so that branch only closes.
  const onBulkDone = useCallback(
    (job: BulkJob) => {
      if (job.kind === 'delete_collection') {
        void onChanged()
        onClose()
        return
      }
      refreshAll().catch((cause: unknown) => setError(errorText(cause)))
    },
    [onChanged, onClose, refreshAll, setError],
  )
  const bulk = useBulkJob(onBulkDone, setError)

  const startBulk = (start: () => Promise<BulkStarted>): void => {
    setError(null)
    bulk.start(start).catch((cause: unknown) => setError(errorText(cause)))
  }

  const candidates = useMemo(() => candidateDocuments(imported, members), [imported, members])
  const needle = needleOf(filter)
  const shownMembers = useMemo(
    () => members.filter((member) => matchesText(needle, member.document.name, member.document.description)),
    [members, needle],
  )
  const shownCandidates = useMemo(() => candidates.filter((doc) => matchesText(needle, doc.name, doc.description)), [candidates, needle])

  // Nothing to show until the collection answers — except why it did not, for a name that is
  // in the hash but not in the home any more.
  if (info === null) return error === null ? null : <p className="muted collection-error">{error}</p>

  const index = info.index
  const facts: [string, string | number][] = [
    ['documents', info.counts.total],
    ['indexed', info.counts.indexed],
    ['active', info.counts.active],
    ['errors', info.counts.error],
    ['rows', index === null ? '—' : index.num_rows],
    ['fragments', index === null ? '—' : index.num_fragments],
  ]

  return (
    <>
      <Tabs tabs={TABS} selected={tab} onSelect={setTab} />

      <div id={TABS[0].id} role="tabpanel" className="modal-panel collection-panel" hidden={tab !== TABS[0].id}>
        <SearchBox value={filter} onChange={setFilter} placeholder="Search documents" />
        <div className="split">
          <section className="pane">
            <span className="pane-head mono muted">In collection · {shownMembers.length}</span>
            <div className="pane-body">
              <ul className="list">
                {shownMembers.map((member) => {
                  const Icon = documentIcon(member.document.suffix)
                  return (
                    <li className="list-item" key={member.document.name}>
                      <Icon className="icon" />
                      <span className="list-text">
                        {member.document.name}
                        <span className="sub">{member.error === null ? member.status : `${member.status} · ${member.error}`}</span>
                      </span>
                      <button
                        className="btn btn-ghost"
                        type="button"
                        aria-label="Remove"
                        disabled={busy}
                        onClick={() => void run(() => api.detachDocument(name, member.document.name))}
                      >
                        <Minus className="icon" />
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          </section>
          <section className="pane">
            <span className="pane-head mono muted">Available · {shownCandidates.length}</span>
            <div className="pane-body">
              <ul className="list">
                {shownCandidates.map((doc) => {
                  const Icon = documentIcon(doc.suffix)
                  return (
                    <li className="list-item" key={doc.name}>
                      <Icon className="icon" />
                      <span className="list-text">
                        {doc.name}
                        <span className="sub">{doc.description || 'No description'}</span>
                      </span>
                      <button
                        className="btn btn-ghost"
                        type="button"
                        aria-label="Add"
                        disabled={busy}
                        onClick={() => void run(() => api.attachDocument(name, doc.name))}
                      >
                        <Plus className="icon" />
                      </button>
                    </li>
                  )
                })}
              </ul>
            </div>
          </section>
        </div>
      </div>

      <div id={TABS[1].id} role="tabpanel" className="collection-panel" hidden={tab !== TABS[1].id}>
        <SearchPanel run={(query) => api.searchCollection(name, query)} placeholder="Search this collection" />
      </div>

      <div id={TABS[2].id} role="tabpanel" className="collection-panel" hidden={tab !== TABS[2].id}>
        {options !== null && (
          <SettingsForm
            settings={info.settings}
            effective={info.effective}
            searchDefaults={info.search}
            options={options}
            outdated={info.index_outdated}
            busy={busy}
            onSave={(next) => void run(() => api.saveCollectionSettings(name, next))}
          >
            <button className="btn" type="button" disabled={bulk.running} onClick={() => startBulk(() => api.indexCollection(name))}>
              <RefreshCw className="icon" />
              Index all
            </button>
            <button
              className="btn btn-ghost"
              type="button"
              disabled={bulk.running}
              onClick={() => {
                if (window.confirm(`Delete collection "${name}"? Its documents stay; only this index goes.`)) {
                  startBulk(() => api.deleteCollection(name))
                }
              }}
            >
              <Trash2 className="icon" />
              Delete collection
            </button>
            {bulk.job !== null && <BulkStatus job={bulk.job} />}
          </SettingsForm>
        )}
      </div>

      <div id={TABS[3].id} role="tabpanel" className="modal-panel collection-panel" hidden={tab !== TABS[3].id}>
        <div className="split">
          <section className="pane">
            <span className="pane-head mono muted">Details</span>
            <div className="pane-body">
              <Kv rows={facts} />
            </div>
          </section>
          <section className="pane">
            <span className="pane-head mono muted">Description</span>
            <div className="pane-body">
              <DescriptionBox
                value={info.description}
                placeholder="What this collection holds"
                onSave={(next) => void run(() => api.describeCollection(name, next))}
              />
            </div>
          </section>
        </div>
      </div>

      {/* One line for whatever the modal last failed at: the panels are tall, and a message
          beside the control that failed would be scrolled out of sight as often as not. */}
      {error !== null && <p className="muted collection-error">{error}</p>}
    </>
  )
}

/** How a queued bulk job is going, beside the button that started it. */
function BulkStatus({ job }: { job: BulkJob }) {
  const running = ACTIVE_JOB_STATUSES.has(job.status)
  const what = job.kind === 'index_collection' ? 'queueing documents' : 'deleting'
  return (
    <span className="muted">
      {running ? `${what}…` : `${what}: ${job.status.toLowerCase()}`}
      {job.progress !== null && ` ${job.progress.done}/${job.progress.total}`}
      {job.error !== null && ` — ${job.error}`}
    </span>
  )
}
