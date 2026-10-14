import { Bot, FilePlus, ListChecks, Minus, Pencil, Plus, Search, type LucideIcon } from 'lucide-react'
import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { api, type SessionAction, type SessionEvent, type SessionSummary } from '../api'
import type { PageProps } from '../App'
import { navigate, type Route } from '../router'
import { Field, GallerySection, Modal, ModalStatus, SearchBox, Shell, Tabs, Tile, type TabDef } from '../ui'
import './Sessions.css'
import { errorText, matchesText, needleOf, relative } from '../format'
import { groupByStatus, tileSub } from './sessions/group'
import { historySub, matchesSearch, searchLines } from './sessions/history'

const ACTION_ICONS: Record<SessionAction, LucideIcon> = {
  search: Search,
  import: FilePlus,
  attach: Plus,
  detach: Minus,
  describe: Pencil,
  collections: ListChecks,
}

/** A session is the set of collections an agent searches under one id: a gallery banded by activity, one modal per session. */
export function Sessions({ route, counts }: PageProps<Extract<Route, { name: 'sessions' }>>) {
  const [loaded, setLoaded] = useState<{ sessions: SessionSummary[]; now: number }>({ sessions: [], now: 0 })
  const [search, setSearch] = useState('')
  const [creating, setCreating] = useState(false)
  const [newId, setNewId] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)

  // "2 min ago" and the Active band are measured from the moment the list was read
  const refresh = useCallback(() => api.sessions().then((sessions) => setLoaded({ sessions, now: Date.now() / 1000 })), [])
  // Read again as a session opens, so its search count is as fresh as the history it reads.
  useEffect(() => {
    refresh().catch((cause: unknown) => setError(errorText(cause)))
  }, [refresh, route.session])

  const startCreating = () => {
    setNewId('')
    setCreateError(null)
    setCreating(true)
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    const id = newId.trim()
    try {
      await api.saveSession(id, [])
      setCreating(false)
      navigate({ name: 'sessions', session: id })
    } catch (cause) {
      setCreateError(errorText(cause))
    }
  }

  // Client-side over the rows already loaded: the API returns every session at once.
  const needle = needleOf(search)
  const groups = groupByStatus(
    loaded.sessions.filter((one) => matchesText(needle, one.id)),
    loaded.now,
  )
  const searches = loaded.sessions.find((one) => one.id === route.session)?.searches
  const close = useCallback(() => navigate({ name: 'sessions' }), [])

  return (
    <Shell current={route.name} counts={counts}>
      <div className="gallery-sections sections">
        <SearchBox id="search" value={search} onChange={setSearch} placeholder="Search sessions" />
        {error !== null && <p className="muted">{error}</p>}
        <GallerySection label="New" large>
          <Tile icon={Plus} name="New session" sub="Create one" hint="Name it; agents search under that id." add onClick={startCreating} />
        </GallerySection>
        {groups.map((group) => (
          <GallerySection key={group.label} label={`${group.label} · ${group.items.length}`} large>
            {group.items.map((one) => (
              <Tile
                key={one.id}
                icon={Bot}
                name={one.id}
                sub={tileSub(one, loaded.now)}
                hint={one.collections.length === 0 ? 'Searches nothing yet' : one.collections.join(' · ')}
                onClick={() => navigate({ name: 'sessions', session: one.id })}
              />
            ))}
          </GallerySection>
        ))}
      </div>
      <SessionModal id={route.session} searches={searches} onClose={close} />
      <Modal open={creating} onClose={() => setCreating(false)} title="New session" subtitle="session">
        {/* A form, so Enter creates the way the browser already does it. */}
        <form className="session-panel modal-scroll" onSubmit={create}>
          <Field label="Session id">
            <input className="input" value={newId} placeholder="Session id" autoFocus onChange={(event) => setNewId(event.target.value)} />
          </Field>
          <div className="row">
            <button className="btn btn-primary" type="submit" disabled={newId.trim() === ''}>
              Create
            </button>
          </div>
          {createError !== null && <ModalStatus tone="error">{createError}</ModalStatus>}
          <ModalStatus>
            Agents call <span className="code">set_session_collections</span> then <span className="code">search</span> with the same id.
          </ModalStatus>
        </form>
      </Modal>
    </Shell>
  )
}

function SessionModal({ id, searches, onClose }: { id: string | undefined; searches: number | undefined; onClose: () => void }) {
  return (
    <Modal open={id !== undefined} onClose={onClose} title={id ?? ''} subtitle="session">
      {id !== undefined && <SessionBody key={id} id={id} searches={searches} />}
    </Modal>
  )
}

/** A session's history as read at one instant: "2 min ago" is measured from it, so every row shares it. */
type LoadedHistory = { events: SessionEvent[]; now: number }

const HISTORY_TAB = 'session-history'

/** `searches` counts every search, where the history holds only the newest events. */
function SessionBody({ id, searches }: { id: string; searches: number | undefined }) {
  const [history, setHistory] = useState<LoadedHistory | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .sessionHistory(id)
      .then((events) => setHistory({ events, now: Date.now() / 1000 }))
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [id])

  // One tab: the strip stays as the panel's heading and carries the count.
  const tab: TabDef = { id: HISTORY_TAB, label: 'Search history', count: searches }

  return (
    <>
      <Tabs tabs={[tab]} selected={HISTORY_TAB} onSelect={() => {}} />
      <div id={HISTORY_TAB} role="tabpanel" className="session-panel">
        {error !== null ? <ModalStatus tone="error">{error}</ModalStatus> : history !== null && <History loaded={history} />}
      </div>
    </>
  )
}

/** A search's head: the tool that ran it, then its context and questions, each after its tag. */
function SearchLines({ event }: { event: SessionEvent }) {
  return (
    <>
      {event.detail.scope && <span className="label-mono muted">{event.detail.scope} search</span>}
      {searchLines(event).map(({ tag, text, context }) => (
        <span className="session-line" key={tag}>
          <span className="descriptor">{tag}</span>
          <span className={context ? 'sub session-query' : 'session-query'}>{text}</span>
        </span>
      ))}
    </>
  )
}

/** What the session did, newest first: each search, import, attach and selection, and what came of it.
 *  A filter narrows it to the searches whose context or questions hold the text. */
function History({ loaded }: { loaded: LoadedHistory }) {
  const [filter, setFilter] = useState('')

  if (loaded.events.length === 0) return <ModalStatus>Nothing yet.</ModalStatus>
  const needle = needleOf(filter)
  const shown = loaded.events.filter((event) => matchesSearch(needle, event))
  return (
    <>
      <SearchBox value={filter} onChange={setFilter} placeholder="Filter by context or question" />
      {shown.length === 0 && <ModalStatus>No search matches.</ModalStatus>}
      <ul className="list">
        {shown.map((event) => {
          const Icon = ACTION_ICONS[event.action]
          const docs = event.detail.documents ?? []
          return (
            <li className="list-item" key={`${event.ts}-${event.action}-${event.subject}`}>
              <Icon className="icon" />
              <span className="list-text">
                {event.action === 'search' ? <SearchLines event={event} /> : <span className="session-query">{event.subject}</span>}
                <span className="sub">
                  {historySub(event)} · {relative(event.ts, loaded.now)}
                </span>
                {docs.length > 0 && <span className="sub session-docs">{docs.join(' · ')}</span>}
              </span>
            </li>
          )
        })}
      </ul>
    </>
  )
}
