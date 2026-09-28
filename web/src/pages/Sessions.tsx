import { Bot, FilePlus, Library, ListChecks, Minus, Pencil, Plus, Search, type LucideIcon } from 'lucide-react'
import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type SessionAction, type SessionEvent, type SessionSummary } from '../api'
import type { PageProps } from '../App'
import { navigate, type Route } from '../router'
import { Check, Field, GallerySection, Modal, SearchBox, SearchPanel, Shell, Tabs, Tile, type TabDef } from '../ui'
import './Sessions.css'
import { errorText, matchesText, needleOf, relative } from '../format'
import { groupByStatus, tileSub } from './sessions/group'
import { historySub } from './sessions/history'

const ACTION_ICONS: Record<SessionAction, LucideIcon> = {
  search: Search,
  import: FilePlus,
  attach: Plus,
  detach: Minus,
  describe: Pencil,
  collections: ListChecks,
}

const TABS: TabDef[] = [
  { id: 'session-collections', label: 'Collections' },
  { id: 'session-search', label: 'Search' },
  { id: 'session-history', label: 'History' },
]

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
  useEffect(() => {
    refresh().catch((cause: unknown) => setError(errorText(cause)))
  }, [refresh])

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
      await refresh()
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
  const chosen = loaded.sessions.find((one) => one.id === route.session)?.collections ?? []
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
      <SessionModal id={route.session} chosen={chosen} onClose={close} onSaved={refresh} />
      <Modal open={creating} onClose={() => setCreating(false)} title="New session" subtitle="session">
        {/* A form, so Enter creates the way the browser already does it. */}
        <form className="session-panel" onSubmit={create}>
          <Field label="Session id">
            <input className="input" value={newId} placeholder="Session id" autoFocus onChange={(event) => setNewId(event.target.value)} />
          </Field>
          <div className="row">
            <button className="btn btn-primary" type="submit" disabled={newId.trim() === ''}>
              Create
            </button>
          </div>
          {createError !== null && <p className="muted session-error">{createError}</p>}
          <p className="muted">
            Agents call <span className="code">set_session_collections</span> then <span className="code">search</span> with the same id.
          </p>
        </form>
      </Modal>
    </Shell>
  )
}

function SessionModal({
  id,
  chosen,
  onClose,
  onSaved,
}: {
  id: string | undefined
  chosen: string[]
  onClose: () => void
  onSaved: () => Promise<unknown>
}) {
  return (
    <Modal open={id !== undefined} onClose={onClose} title={id ?? ''} subtitle="session">
      {id !== undefined && <SessionBody key={id} id={id} chosen={chosen} onSaved={onSaved} />}
    </Modal>
  )
}

function SessionBody({ id, chosen, onSaved }: { id: string; chosen: string[]; onSaved: () => Promise<unknown> }) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [tab, setTab] = useState<string>(TABS[0].id)
  const [error, setError] = useState<string | null>(null)
  const [searched, setSearched] = useState(0) // bumps after each search here, so the history re-reads

  useEffect(() => {
    api
      .collections({ page_size: MAX_PAGE_SIZE, sort: 'name' })
      .then((page) => setCollections(page.items))
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [])

  // Every change is saved at once: a session is one small list, and the agent may read it back
  // between two clicks. The kept order is the caller's; a new pick goes to the end.
  const toggle = (name: string, on: boolean): void => {
    const next = on ? [...chosen, name] : chosen.filter((one) => one !== name)
    setError(null)
    api
      .saveSession(id, next)
      .then(() => onSaved())
      .catch((cause: unknown) => setError(errorText(cause)))
  }

  const run = async (query: string) => {
    const found = await api.explore(query, 'chunk', { session_id: id })
    setSearched((count) => count + 1)
    return found
  }

  return (
    <>
      <Tabs tabs={TABS} selected={tab} onSelect={setTab} />

      <div id={TABS[0].id} role="tabpanel" className="session-panel" hidden={tab !== TABS[0].id}>
        {error !== null && <p className="muted">{error}</p>}
        <ul className="list">
          {collections.map((collection) => (
            <li className="list-item" key={collection.name}>
              <Library className="icon" />
              <span className="list-text">
                <Check
                  label={collection.name}
                  checked={chosen.includes(collection.name)}
                  onChange={(on) => toggle(collection.name, on)}
                />
                <span className="sub">{collection.counts.total} documents</span>
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div id={TABS[1].id} role="tabpanel" className="session-panel" hidden={tab !== TABS[1].id}>
        <SearchPanel run={run} placeholder="Search this session" plural="chunks" />
      </div>

      <div id={TABS[2].id} role="tabpanel" className="session-panel" hidden={tab !== TABS[2].id}>
        {tab === TABS[2].id && <History id={id} version={searched} />}
      </div>
    </>
  )
}

/** What the session did, newest first: each search, import, attach and selection, and what came of it. */
function History({ id, version }: { id: string; version: number }) {
  const [loaded, setLoaded] = useState<{ events: SessionEvent[]; now: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // "2 min ago" is measured from the moment the list was read, so every row shares one instant
  useEffect(() => {
    api
      .sessionHistory(id)
      .then((events) => setLoaded({ events, now: Date.now() / 1000 }))
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [id, version])

  if (error !== null) return <p className="muted">{error}</p>
  if (loaded === null) return null
  if (loaded.events.length === 0) return <p className="muted">Nothing yet.</p>
  return (
    <ul className="list">
      {loaded.events.map((event) => {
        const Icon = ACTION_ICONS[event.action]
        const docs = event.detail.documents ?? []
        return (
          <li className="list-item" key={`${event.ts}-${event.action}-${event.subject}`}>
            <Icon className="icon" />
            <span className="list-text">
              <span className="session-query">{event.subject}</span>
              <span className="sub">
                {historySub(event)} · {relative(event.ts, loaded.now)}
              </span>
              {docs.length > 0 && <span className="sub session-docs">{docs.join(' · ')}</span>}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
