import { useEffect, useState } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type Granularity, type Hit, type Passage, type SessionSummary, type Source } from '../api'
import type { PageProps } from '../App'
import { HitGrid, MatchModal, Picker, SearchBox, SearchTook, Shell, Tabs, type Match, type PickerOption, type TabDef } from '../ui'
import './Explore.css'
import { ALL_SCOPE, parseScope, scopeParams, SESSION_PREFIX } from './explore/scope'
import { errorText } from '../format'

const MATCHES_TAB = 'tab-matches'
const SOURCES_TAB = 'tab-sources'

// What each granularity is called on screen, and what its results are: the second picker's
// options and the first tab's label come from here.
const GRANULARITIES: Array<PickerOption<Granularity> & { plural: string }> = [
  { value: 'chunk', label: 'Chunks', plural: 'chunks', sub: 'as indexed' },
  { value: 'passage', label: 'Passages', plural: 'passages', sub: 'adjacent chunks, widened' },
  { value: 'excerpt', label: 'Excerpts', plural: 'excerpts', sub: 'what an agent reads' },
]
const granularityOf = (value: Granularity) => GRANULARITIES.find((one) => one.value === value) ?? GRANULARITIES[0]

/** Search across collections: the matches at one granularity on one tab, the sources on the other. */
export function Explore({ route, counts }: PageProps) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [scope, setScope] = useState(ALL_SCOPE)
  const [granularity, setGranularity] = useState<Granularity>('excerpt')
  const [query, setQuery] = useState('')
  const [ran, setRan] = useState('') // the query the results on screen answer
  const [ranAs, setRanAs] = useState<Granularity>('excerpt') // and the granularity it ran at
  const [matches, setMatches] = useState<Array<Hit | Passage>>([])
  const [sources, setSources] = useState<Source[]>([])
  const [tab, setTab] = useState(MATCHES_TAB)
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

  const scopes: PickerOption<string>[] = [
    { value: ALL_SCOPE, label: 'All collections', sub: `${counts.documents ?? 0} documents` },
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
    { id: MATCHES_TAB, label: `${granularityOf(ranAs).label} · ${matches.length}` },
    { id: SOURCES_TAB, label: `Sources · ${sources.length}` },
  ]

  const submit = async () => {
    const text = query.trim()
    if (text === '') return
    const where = scopeParams(parseScope(scope))
    setError(null)
    setBusy(true)
    const started = performance.now()
    try {
      // Both tabs answer the same query over the same scope, so both requests go out together.
      const [found, named] = await Promise.all([api.explore(text, granularity, where), api.searchSources(text, where)])
      setMatches(found)
      setSources(named.documents)
      setRan(text)
      setRanAs(granularity)
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
    setMatches([])
    setSources([])
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
          placeholder="Search your collections"
          onSubmit={() => void submit()}
          onClear={clear}
          busy={busy}
          scope={
            <>
              <Picker options={scopes} value={scope} onChange={setScope} ariaLabel="Search scope" />
              <Picker options={GRANULARITIES} value={granularity} onChange={setGranularity} ariaLabel="Result granularity" />
            </>
          }
        />
        {error !== null && <p className="muted">{error}</p>}
        <SearchTook counts={`${matches.length} ${granularityOf(ranAs).plural} · ${sources.length} sources`} ms={took} />
        <Tabs tabs={tabs} selected={tab} onSelect={setTab} />
        <div id={MATCHES_TAB} role="tabpanel" hidden={tab !== MATCHES_TAB}>
          <HitGrid results={matches} query={ran} onOpen={setOpen} />
        </div>
        <div id={SOURCES_TAB} role="tabpanel" hidden={tab !== SOURCES_TAB}>
          <HitGrid results={sources} query={ran} onOpen={setOpen} />
        </div>
      </div>
      <MatchModal match={open} query={ran} onClose={() => setOpen(null)} />
    </Shell>
  )
}
