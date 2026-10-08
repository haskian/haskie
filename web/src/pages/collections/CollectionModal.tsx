import { Minus, Plus, RefreshCw, Sparkles, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  api,
  MAX_PAGE_SIZE,
  type BulkKind,
  type BulkStarted,
  type CollectionInfo,
  type Document,
  type Member,
  type OperationProgress,
  type Options,
} from '../../api'
import { errorText, matchesText, needleOf } from '../../format'
import { useOperation } from '../../hooks/useOperation'
import { usePoll } from '../../hooks/usePoll'
import { useRun } from '../../hooks/useRun'
import { BulkStatus, BusyButton, DescriptionBox, documentIcon, Kv, ListPane, Modal, ModalStatus, RenameForm, SearchBox, SearchPanel, Tabs, type TabDef } from '../../ui'
import { plural } from '../../ui/match'
import { candidateDocuments, otherCollections } from './candidates'
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
  onRenamed,
}: {
  name: string | undefined
  onClose: () => void
  onChanged: () => Promise<unknown>
  onRenamed: (to: string) => void
}) {
  return (
    <Modal open={name !== undefined} onClose={onClose} title={name ?? ''} subtitle="collection">
      {name !== undefined && <CollectionBody key={name} name={name} onClose={onClose} onChanged={onChanged} onRenamed={onRenamed} />}
    </Modal>
  )
}

function CollectionBody({
  name,
  onClose,
  onChanged,
  onRenamed,
}: {
  name: string
  onClose: () => void
  onChanged: () => Promise<unknown>
  onRenamed: (to: string) => void
}) {
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
  // A document may be attached only once it is imported, so the other pane lists exactly those,
  // newest first: the one just imported is usually the one to add.
  const refreshImported = useCallback(
    () =>
      api
        .documents({ status: 'imported', page_size: MAX_PAGE_SIZE, sort: 'created_at', order: 'desc' })
        .then((page) => setImported(page.items)),
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

  // "Index all", "Describe with AI" and "Delete collection" are accepted (202) and run in the
  // background, so the modal follows the operation. The callbacks keep one identity, or `useOperation` rebuilds its
  // poll on every tick. A deleted collection has nothing left to re-read, so that branch closes.
  const onBulkDone = useCallback(
    (operation: OperationProgress) => {
      if (operation.kind === 'delete_collection' && operation.status === 'SUCCESS') {
        void onChanged()
        onClose()
        return
      }
      refreshAll().catch((cause: unknown) => setError(errorText(cause)))
    },
    [onChanged, onClose, refreshAll, setError],
  )
  const bulk = useOperation(onBulkDone, setError)

  const startBulk = (kind: BulkKind, start: () => Promise<BulkStarted>): void => {
    setError(null)
    bulk.start(kind, start).catch((cause: unknown) => setError(errorText(cause)))
  }

  // Not through `run`: its re-read would ask for the old name. The route moves to the new one,
  // and the modal remounts there.
  const rename = (to: string): void => {
    setError(null)
    api
      .renameCollection(name, to)
      .then(async (renamed) => {
        await onChanged()
        onRenamed(renamed.name)
      })
      .catch((cause: unknown) => setError(errorText(cause)))
  }

  const candidates = useMemo(() => candidateDocuments(imported, members), [imported, members])
  const elsewhere = useMemo(() => otherCollections(imported, name), [imported, name])
  const needle = needleOf(filter)
  const shownMembers = useMemo(
    () => members.filter((member) => matchesText(needle, member.document.name, member.document.description)),
    [members, needle],
  )
  const shownCandidates = useMemo(() => candidates.filter((doc) => matchesText(needle, doc.name, doc.description)), [candidates, needle])

  // Nothing to show until the collection answers. The one exception is why it did not answer,
  // for a name that is in the hash but not in the home any more.
  if (info === null) return error === null ? null : <ModalStatus tone="error">{error}</ModalStatus>

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
          <ListPane head="In collection" count={shownMembers.length}>
            {shownMembers.map((member) => {
              const Icon = documentIcon(member.document.suffix)
              return (
                <li className="list-item" key={member.document.name}>
                  <Icon className="icon" />
                  <span className="list-text">
                    <DocumentName document={member.document} />
                    <span className="sub">{member.error === null ? member.status : `${member.status} · ${member.error}`}</span>
                  </span>
                  <AlsoIn collections={elsewhere.get(member.document.name) ?? []} />
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
          </ListPane>
          <ListPane head="Available" count={shownCandidates.length}>
            {shownCandidates.map((doc) => {
              const Icon = documentIcon(doc.suffix)
              return (
                <li className="list-item" key={doc.name}>
                  <Icon className="icon" />
                  <span className="list-text">
                    <DocumentName document={doc} />
                    {doc.pages != null && <span className="sub">{plural(doc.pages, 'page')}</span>}
                  </span>
                  <AlsoIn collections={doc.collections} />
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
          </ListPane>
        </div>
      </div>

      <div id={TABS[1].id} role="tabpanel" className="collection-panel" hidden={tab !== TABS[1].id}>
        <SearchPanel active={tab === TABS[1].id} run={(query) => api.explore(query, 'passage', { collections: [name] })} placeholder="Search this collection" plural="passages" />
      </div>

      <div id={TABS[2].id} role="tabpanel" className="collection-panel" hidden={tab !== TABS[2].id}>
        {options !== null && (
          <SettingsForm
            overrides={info.overrides}
            effective={info.effective}
            searchDefaults={info.search}
            options={options}
            outdated={info.index_outdated}
            active={tab === TABS[2].id}
            busy={busy}
            onSave={(next) => void run(() => api.saveCollectionOverrides(name, next))}
          >
            <BusyButton
              busy={bulk.running && bulk.kind === 'index_collection'}
              busyLabel="Queueing documents ..."
              progress={bulk.operation?.kind === 'index_collection' ? bulk.operation.progress : null}
              className="btn"
              type="button"
              disabled={bulk.running}
              onClick={() => startBulk('index_collection', () => api.indexCollection(name))}
            >
              <RefreshCw className="icon" />
              Index all
            </BusyButton>
          </SettingsForm>
        )}
      </div>

      <div id={TABS[3].id} role="tabpanel" className="modal-panel collection-panel" hidden={tab !== TABS[3].id}>
        <RenameForm key={name} name={name} label="Collection name" busy={bulk.running} onRename={rename} />
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
              {/* Keyed by the text: the box is uncontrolled, and a written description replaces
                  what it shows. */}
              <DescriptionBox
                key={info.description}
                value={info.description}
                placeholder="What this collection holds"
                onSave={(next) => void run(() => api.describeCollection(name, next))}
              />
              <div className="row">
                <BusyButton
                  busy={bulk.running && bulk.kind === 'summarize_collection'}
                  busyLabel="Describing ..."
                  progress={bulk.operation?.kind === 'summarize_collection' ? bulk.operation.progress : null}
                  className="btn"
                  type="button"
                  disabled={bulk.running}
                  onClick={() => {
                    const replace = info.description === '' || window.confirm('Replace the description with one the AI writes from its documents?')
                    if (replace) startBulk('summarize_collection', () => api.generateCollectionDescription(name))
                  }}
                >
                  <Sparkles className="icon" />
                  Describe with AI
                </BusyButton>
              </div>
            </div>
          </section>
        </div>
        <div className="row row-loose">
          <BusyButton
            busy={bulk.running && bulk.kind === 'delete_collection'}
            busyLabel="Deleting ..."
            progress={bulk.operation?.kind === 'delete_collection' ? bulk.operation.progress : null}
            className="btn btn-ghost"
            type="button"
            disabled={bulk.running}
            onClick={() => {
              if (window.confirm(`Delete collection "${name}"? Its documents stay; only this index goes.`)) {
                startBulk('delete_collection', () => api.deleteCollection(name))
              }
            }}
          >
            <Trash2 className="icon" />
            Delete collection
          </BusyButton>
        </div>
      </div>

      {bulk.operation !== null && <BulkStatus operation={bulk.operation} />}
      {error !== null && <ModalStatus tone="error">{error}</ModalStatus>}
    </>
  )
}

/** A document's name, which shows its description on hover, as its tile does on the documents
 *  page; nothing shows when it has none. */
function DocumentName({ document }: { document: Pick<Document, 'name' | 'description'> }) {
  return (
    <span className="document-name">
      {document.name}
      {document.description && (
        <span className="hint hint-below" role="tooltip">
          {document.description}
        </span>
      )}
    </span>
  )
}

/** The other collections that hold a document, as a tag naming them on hover; nothing for none. */
function AlsoIn({ collections }: { collections: string[] }) {
  if (collections.length === 0) return null
  return (
    <span className="also-in" tabIndex={0}>
      <span className="tag">
        <span className="kind">also in</span>
        <span>{collections.length}</span>
      </span>
      {/* flipped: the tag sits at the pane's right edge */}
      <span className="hint hint-below flip" role="tooltip">
        {collections.map((collection) => (
          <span key={collection}>{collection}</span>
        ))}
      </span>
    </span>
  )
}
