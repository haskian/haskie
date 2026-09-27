import { Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type Granularity, type ScoreStep, type SearchScope, type SessionSummary, type StepTiming } from '../api'
import type { PageProps } from '../App'
import { HitGrid, MatchModal, Picker, SearchBox, SearchTook, Shell, type Match, type PickerOption } from '../ui'
import './Explore.css'
import { MAX_ASPECTS, questionsOf } from './explore/questions'
import { ALL_SCOPE, parseScope, scopeParams, SESSION_PREFIX } from './explore/scope'
import { errorText } from '../format'

/** What a search answers with: a granularity of matches, the excerpts an agent reads, or the
 *  documents that hold them. */
type Answer = Granularity | 'excerpt' | 'source'

// What each answer is called on screen, and what its results are: the second picker's options.
const ANSWERS: Array<PickerOption<Answer> & { plural: string }> = [
  { value: 'chunk', label: 'Chunks', plural: 'chunks', sub: 'as indexed' },
  { value: 'passage', label: 'Passages', plural: 'passages', sub: 'adjacent chunks of a section' },
  { value: 'excerpt', label: 'Excerpts', plural: 'excerpts', sub: 'what an agent reads' },
  { value: 'source', label: 'Sources', plural: 'sources', sub: 'the documents that answer it' },
]
const answerOf = (value: Answer) => ANSWERS.find((one) => one.value === value) ?? ANSWERS[0]

/** How an excerpts search is asked: one question, or several aspects under a shared context. */
type Asking = 'single' | 'multi'

const ASKINGS: PickerOption<Asking>[] = [
  { value: 'single', label: 'Single query' },
  { value: 'multi', label: 'Multi-aspect query' },
]

/** What one search brings back: its results, how long each step took, and what the excerpts
 *  say they lack (only excerpts say): the words of the questions they never hold, and the
 *  questions they do not answer. */
interface Found {
  results: Match[]
  steps: StepTiming[]
  scoring: ScoreStep[] // how their scores came to be
  missing: string[]
  uncovered: string[]
}

/** The one request an answer takes: its own route for excerpts and for documents, the explore
 *  route for chunks and passages. Excerpts take every question asked and the shared background;
 *  the others take the first question. */
async function search(questions: string[], context: string, answer: Answer, where: SearchScope): Promise<Found> {
  const [text] = questions
  if (answer === 'source') {
    const found = await api.searchSources(text, where)
    return { results: found.body.documents, steps: found.steps, scoring: found.scoring, missing: [], uncovered: [] }
  }
  if (answer === 'excerpt') {
    const found = await api.searchExcerpts(questions, where, context.trim() || undefined)
    const { excerpts, missing_terms, uncovered } = found.body
    return { results: excerpts, steps: found.steps, scoring: found.scoring, missing: missing_terms, uncovered }
  }
  const found = await api.explore(text, answer, where)
  return { results: found.body, steps: found.steps, scoring: found.scoring, missing: [], uncovered: [] }
}

/** Search across collections, answering at the granularity the picker names: matches, or the
 *  documents they come from. One request a search. */
export function Explore({ route, counts }: PageProps) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [scope, setScope] = useState(ALL_SCOPE)
  const [answer, setAnswer] = useState<Answer>('excerpt')
  const [asking, setAsking] = useState<Asking>('single')
  const [aspects, setAspects] = useState(['']) // the query first, then the other aspects
  const [ran, setRan] = useState('') // the query the results on screen answer
  const [ranAs, setRanAs] = useState<Answer>('excerpt') // and what it answered with
  const [results, setResults] = useState<Match[]>([])
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [busy, setBusy] = useState(false)
  const [took, setTook] = useState<number | null>(null) // ms the last query took
  const [steps, setSteps] = useState<StepTiming[]>([]) // and how long each of its steps took
  const [scoring, setScoring] = useState<ScoreStep[]>([]) // and how its scores came to be
  const [missing, setMissing] = useState<string[]>([]) // words the excerpts on screen lack
  const [uncovered, setUncovered] = useState<string[]>([]) // and the questions they do not answer
  const [context, setContext] = useState('') // multi-aspect only: the background every aspect shares
  const [asked, setAsked] = useState<string[]>([]) // the questions the results on screen answer

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

  const multi = answer === 'excerpt' && asking === 'multi'

  const submit = async () => {
    const questions = questionsOf(multi ? aspects : aspects.slice(0, 1))
    if (questions.length === 0) return
    setError(null)
    setBusy(true)
    const started = performance.now()
    try {
      const found = await search(questions, multi ? context : '', answer, scopeParams(parseScope(scope)))
      setResults(found.results)
      setSteps(found.steps)
      setScoring(found.scoring)
      setMissing(found.missing)
      setUncovered(found.uncovered)
      setAsked(questions)
      setRan(questions[0])
      setRanAs(answer)
      setTook(Math.round(performance.now() - started))
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  const setAspect = (at: number, text: string) => setAspects((all) => all.map((one, i) => (i === at ? text : one)))

  const clear = () => {
    setAspect(0, '')
    setRan('')
    setResults([])
    setSteps([])
    setScoring([])
    setMissing([])
    setUncovered([])
    setAsked([])
    setError(null)
    setTook(null)
  }

  const reset = () => {
    setAspects([''])
    setContext('')
  }

  const pickers = (
    <>
      <Picker options={scopes} value={scope} onChange={setScope} ariaLabel="Search scope" />
      <Picker options={ANSWERS} value={answer} onChange={setAnswer} ariaLabel="Result granularity" />
    </>
  )
  return (
    <Shell current={route.name} counts={counts}>
      <div className="explore">
        {answer === 'excerpt' && (
          <div className="explore-bar">
            <div className="input-group">
              {pickers}
              <Picker options={ASKINGS} value={asking} onChange={setAsking} ariaLabel="Query kind" />
            </div>
            {multi && (
              <button className="btn" type="button" onClick={reset}>
                Reset
              </button>
            )}
          </div>
        )}
        {multi ? (
          <div className="explore-aspects">
            <form onSubmit={(event) => { event.preventDefault(); void submit() }}>
              <input className="input" aria-label="Context" placeholder="Context the aspects share: it steers what they mean, never the words they match" maxLength={200} value={context} onChange={(event) => setContext(event.target.value)} />
            </form>
            {aspects.map((text, at) => (
              <SearchBox
                key={at}
                id={at === 0 ? 'q' : undefined}
                value={text}
                onChange={(next) => setAspect(at, next)}
                placeholder={at === 0 ? 'Search your collections' : 'Another aspect, searched on its own'}
                onSubmit={() => void submit()}
                onClear={at === 0 ? clear : undefined}
                busy={at === 0 && busy}
                scope={<span className="mono aspect-label">Q{at + 1}</span>}
              />
            ))}
            <button className="btn" type="button" aria-label="Add an aspect" disabled={aspects.length >= MAX_ASPECTS} onClick={() => setAspects((all) => [...all, ''])}>
              <Plus className="icon" />
            </button>
          </div>
        ) : (
          <SearchBox
            id="q"
            value={aspects[0]}
            onChange={(text) => setAspect(0, text)}
            placeholder="Search your collections"
            onSubmit={() => void submit()}
            onClear={clear}
            busy={busy}
            scope={answer === 'excerpt' ? undefined : pickers}
          />
        )}
        {error !== null && <p className="muted">{error}</p>}
        <SearchTook counts={`${results.length} ${answerOf(ranAs).plural}`} ms={took} steps={steps} />
        {missing.length > 0 && <p className="muted">No excerpt says: {missing.join(', ')}</p>}
        {uncovered.length > 0 && <p className="muted">Unanswered: {uncovered.join(' · ')}</p>}
        <HitGrid results={results} query={ran} onOpen={setOpen} questions={asked} />
      </div>
      <MatchModal match={open} query={ran} scoring={scoring} onClose={() => setOpen(null)} />
    </Shell>
  )
}
