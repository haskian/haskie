import { useEffect, useState } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type Granularity, type SearchScope, type SessionSummary, type StepTiming } from '../api'
import type { PageProps } from '../App'
import { HitGrid, MatchModal, Picker, SearchBox, SearchTook, Shell, type Match, type PickerOption } from '../ui'
import './Explore.css'
import { ALL_SCOPE, parseScope, scopeParams, SESSION_PREFIX } from './explore/scope'
import { errorText } from '../format'

/** What a search answers with: a granularity of matches, or the documents that hold them. */
type Answer = Granularity | 'source'

// What each answer is called on screen, and what its results are: the second picker's options.
const ANSWERS: Array<PickerOption<Answer> & { plural: string }> = [
  { value: 'chunk', label: 'Chunks', plural: 'chunks', sub: 'as indexed' },
  { value: 'passage', label: 'Passages', plural: 'passages', sub: 'adjacent chunks, widened' },
  { value: 'excerpt', label: 'Excerpts', plural: 'excerpts', sub: 'what an agent reads' },
  { value: 'source', label: 'Sources', plural: 'sources', sub: 'the documents that answer it' },
]
const answerOf = (value: Answer) => ANSWERS.find((one) => one.value === value) ?? ANSWERS[0]

/** The one request an answer takes: the sources route for documents, the explore route else. */
async function search(text: string, answer: Answer, where: SearchScope): Promise<{ results: Match[]; steps: StepTiming[] }> {
  if (answer === 'source') {
    const found = await api.searchSources(text, where)
    return { results: found.body.documents, steps: found.steps }
  }
  const found = await api.explore(text, answer, where)
  return { results: found.body, steps: found.steps }
}

/** Search across collections, answering at the granularity the picker names: matches, or the
 *  documents they come from. One request a search. */
export function Explore({ route, counts }: PageProps) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [scope, setScope] = useState(ALL_SCOPE)
  const [answer, setAnswer] = useState<Answer>('excerpt')
  const [query, setQuery] = useState('')
  const [ran, setRan] = useState('') // the query the results on screen answer
  const [ranAs, setRanAs] = useState<Answer>('excerpt') // and what it answered with
  const [results, setResults] = useState<Match[]>([])
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [busy, setBusy] = useState(false)
  const [took, setTook] = useState<number | null>(null) // ms the last query took
  const [steps, setSteps] = useState<StepTiming[]>([]) // and how long each of its steps took

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

  const submit = async () => {
    const text = query.trim()
    if (text === '') return
    setError(null)
    setBusy(true)
    const started = performance.now()
    try {
      const found = await search(text, answer, scopeParams(parseScope(scope)))
      setResults(found.results)
      setSteps(found.steps)
      setRan(text)
      setRanAs(answer)
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
    setResults([])
    setSteps([])
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
              <Picker options={ANSWERS} value={answer} onChange={setAnswer} ariaLabel="Result granularity" />
            </>
          }
        />
        {error !== null && <p className="muted">{error}</p>}
        <SearchTook counts={`${results.length} ${answerOf(ranAs).plural}`} ms={took} steps={steps} />
        <HitGrid results={results} query={ran} onOpen={setOpen} />
      </div>
      <MatchModal match={open} query={ran} onClose={() => setOpen(null)} />
    </Shell>
  )
}
