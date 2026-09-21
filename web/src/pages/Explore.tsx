import { useEffect, useState } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type DocumentMatch, type Hit, type SessionSummary } from '../api'
import type { PageProps } from '../App'
import { HitGrid, MatchModal, Picker, SearchBox, SearchTook, Shell, Tabs, type Match, type PickerOption, type TabDef } from '../ui'
import './Explore.css'
import { ALL_SCOPE, parseScope, scopeCollections, searchSections, SESSION_PREFIX } from './explore/scope'
import { errorText } from '../format'

const SECTIONS_TAB = 'tab-sections'
const LITERATURE_TAB = 'tab-literature'

/** Search across collections: passages on one tab, whole documents on the other. */
export function Explore({ route, counts }: PageProps) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [scope, setScope] = useState(ALL_SCOPE)
  const [query, setQuery] = useState('')
  const [ran, setRan] = useState('') // the query the results on screen answer
  const [ranIn, setRanIn] = useState<string[] | undefined>(undefined) // and the collections it ran over
  const [hits, setHits] = useState<Hit[]>([])
  const [matches, setMatches] = useState<DocumentMatch[]>([])
  const [tab, setTab] = useState(SECTIONS_TAB)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [busy, setBusy] = useState(false)
  const [took, setTook] = useState<number | null>(null) // ms the last query took

  useEffect(() => {
    Promise.all([api.collections({ page_size: MAX_PAGE_SIZE }), api.sessions()])
      .then(([page, saved]) => {
        setCollections(page.items)
        setSessions(saved)
      })
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [])

  // ponytail: the design's second picker (Hybrid / Vector / Text / Reranked) is dropped. The API
  // takes no per-query mode; the mode is a collection setting, edited on the Collections page.
  const scopes: PickerOption<string>[] = [
    { value: ALL_SCOPE, label: 'All collections', sub: `${counts.documents ?? 0} documents · full-text` },
    ...collections.map((collection) => ({
      value: collection.name,
      label: collection.name,
      sub: `${collection.counts.total} documents`,
    })),
    ...sessions.map(({ id, collections: names }) => ({
      value: `${SESSION_PREFIX}${id}`,
      label: id,
      sub: `${names.join(', ') || 'no collections'} · session`,
    })),
  ]

  const tabs: TabDef[] = [
    { id: SECTIONS_TAB, label: `Sections · ${hits.length}` },
    { id: LITERATURE_TAB, label: `Literature · ${matches.length}` },
  ]

  const submit = async () => {
    const text = query.trim()
    if (text === '') return
    const picked = parseScope(scope)
    const filter = scopeCollections(picked, Object.fromEntries(sessions.map((one) => [one.id, one.collections])))
    setError(null)
    setBusy(true)
    const started = performance.now()
    try {
      // Both tabs answer the same query, so both requests go out together.
      const [sections, literature] = await Promise.all([
        searchSections(picked, text),
        // A session holding no collections matches nothing, while an absent filter would mean
        // "every collection": the empty scope is answered here instead of by the backend.
        filter?.length === 0 ? Promise.resolve<DocumentMatch[]>([]) : api.searchDocuments(text, filter),
      ])
      setHits(sections)
      setMatches(literature)
      setRan(text)
      setRanIn(filter)
      setTook(Math.round(performance.now() - started))
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  const clear = () => {
    setQuery('')
    setRan('')
    setHits([])
    setMatches([])
    setError(null)
    setTook(null)
  }

  return (
    <Shell current={route.name} counts={counts}>
      <div className="explore">
        <SearchBox
          id="q"
          value={query}
          onChange={setQuery}
          placeholder="Search across collections"
          onSubmit={() => void submit()}
          onClear={clear}
          busy={busy}
          scope={<Picker options={scopes} value={scope} onChange={setScope} ariaLabel="Search scope" />}
        />
        {error !== null && <p className="muted">{error}</p>}
        <SearchTook counts={`${hits.length} sections · ${matches.length} documents`} ms={took} />
        <Tabs tabs={tabs} selected={tab} onSelect={setTab} />
        <div id={SECTIONS_TAB} role="tabpanel" hidden={tab !== SECTIONS_TAB}>
          <HitGrid hits={hits} query={ran} onOpen={setOpen} />
        </div>
        <div id={LITERATURE_TAB} role="tabpanel" hidden={tab !== LITERATURE_TAB}>
          <HitGrid matches={matches} query={ran} onOpen={setOpen} />
        </div>
      </div>
      <MatchModal hit={open} query={ran} collections={ranIn} onClose={() => setOpen(null)} />
    </Shell>
  )
}
