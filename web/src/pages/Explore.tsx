import { Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, MAX_PAGE_SIZE, type CollectionSummary, type Granularity, type MappedDocument, type MappedSection, type ScoreStep, type SearchScope, type SessionSummary, type StepTiming } from '../api'
import type { PageProps } from '../App'
import { HitGrid, MapDocuments, MatchModal, SectionsModal, Picker, SearchBox, SearchTook, SectionGrid, Shell, Toggle, type Match, type OpenedSection, type PickerOption } from '../ui'
import './Explore.css'
import { MAX_ASPECTS, questionsOf } from './explore/questions'
import { ALL_SCOPE, parseScope, scopeParams, SESSION_PREFIX } from './explore/scope'
import { errorText } from '../format'

/** What a search answers with: a granularity of matches, the excerpts an agent reads, or a map
 *  of the sections the topic touches and the documents it reached. */
type Answer = Granularity | 'excerpt' | 'section'

// The answers an agent gets too, marked with the MCP tool that gives it the same.
const mcp = (tool: string) => ({ text: 'MCP', title: `An agent gets the same from the MCP tool ${tool}` })

// What each answer is called on screen, and what its results are: the second picker's options.
const ANSWERS: Array<PickerOption<Answer> & { plural: string }> = [
  { value: 'excerpt', label: 'Excerpts', plural: 'excerpts', sub: 'what an agent reads', tag: mcp('search_excerpts') },
  { value: 'section', label: 'Sections', plural: 'sections', sub: 'a map of the sections and documents it touches', tag: mcp('search_sections') },
  { value: 'passage', label: 'Passages', plural: 'passages', sub: 'adjacent chunks of a section' },
  { value: 'chunk', label: 'Chunks', plural: 'chunks', sub: 'as indexed' },
]
const answerOf = (value: Answer) => ANSWERS.find((one) => one.value === value) ?? ANSWERS[0]

/** How an excerpts search is asked: one question, or several aspects under a shared context. */
type Asking = 'single' | 'multi'

const ASKINGS: PickerOption<Asking>[] = [
  { value: 'single', label: 'Single query' },
  { value: 'multi', label: 'Multi-aspect query' },
]

/** What one search brings back: its results (a map's are its `sections` and `documents`), how
 *  long each step took, and what it says it lacks: the words of the questions no excerpt holds
 *  (only excerpts say), and the questions the sources do not answer or match only weakly. A map
 *  also names the fewest collections that hold every section and document on it. */
interface Found {
  body: unknown // the response as the endpoint sent it, for the debug view
  results: Match[]
  sections: MappedSection[]
  documents: MappedDocument[]
  holders: string[]
  steps: StepTiming[]
  scoring: ScoreStep[] // how their scores came to be
  missing: string[]
  uncovered: string[]
}

/** The search whose results are on screen: what it found, the questions it asked (the first is
 *  the one its matches are marked by), what it answered with, and the ms it took. */
interface Shown extends Found {
  asked: string[]
  context: string // the background the questions shared, as it was sent
  as: Answer
  took: number | null // null before the first search: the line above the results stays blank
}

const NONE = { results: [], sections: [], documents: [], holders: [], missing: [], uncovered: [] }
const NOTHING: Shown = { ...NONE, body: null, steps: [], scoring: [], asked: [], context: '', as: 'excerpt', took: null }

/** The one request an answer takes: its own route for excerpts and sections, the
 *  explore route for chunks and passages. Excerpts take every question asked and the shared
 *  background; the others take the first question. */
async function search(questions: string[], context: string, answer: Answer, where: SearchScope): Promise<Found> {
  const [text] = questions
  if (answer === 'section') {
    const found = await api.searchSections(text, where)
    const { sections, documents, collections, uncovered = [] } = found.body
    return { ...NONE, body: found.body, sections, documents, holders: collections, steps: found.steps, scoring: found.scoring, uncovered }
  }
  if (answer === 'excerpt') {
    const found = await api.searchExcerpts(questions, where, context.trim() || undefined)
    const { excerpts, missing_terms, uncovered } = found.body
    return { ...NONE, body: found.body, results: excerpts, steps: found.steps, scoring: found.scoring, missing: missing_terms, uncovered }
  }
  const found = await api.explore(text, answer, where)
  return { ...NONE, body: found.body, results: found.body, steps: found.steps, scoring: found.scoring }
}

/** Search across collections, answering as the picker names: matches, excerpts, or a map of
 *  sections and documents. One request a search. */
export function Explore({ route, counts }: PageProps) {
  const [collections, setCollections] = useState<CollectionSummary[]>([])
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [scope, setScope] = useState(ALL_SCOPE)
  const [answer, setAnswer] = useState<Answer>('excerpt')
  const [asking, setAsking] = useState<Asking>('single')
  const [aspects, setAspects] = useState(['']) // the query first, then the other aspects
  const [context, setContext] = useState('') // multi-aspect only: the background every aspect shares
  const [shown, setShown] = useState<Shown>(NOTHING)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [opened, setOpened] = useState<OpenedSection | null>(null)
  const [busy, setBusy] = useState(false)
  const [debug, setDebug] = useState(false) // the raw response in place of the results

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
      const shared = multi ? context.trim() : ''
      const found = await search(questions, shared, answer, scopeParams(parseScope(scope)))
      setShown({ ...found, asked: questions, context: shared, as: answer, took: Math.round(performance.now() - started) })
    } catch (cause) {
      setError(errorText(cause))
    } finally {
      setBusy(false)
    }
  }

  const setAspect = (at: number, text: string) => setAspects((all) => all.map((one, i) => (i === at ? text : one)))

  const clear = () => {
    setAspect(0, '')
    setShown(NOTHING)
    setError(null)
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
                onClear={at === 0 ? clear : () => setAspects((all) => all.filter((_, i) => i !== at))}
                clearLabel={at === 0 ? 'Clear' : 'Remove aspect'}
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
        <div className="explore-status">
          <SearchTook counts={`${shown.as === 'section' ? shown.sections.length : shown.results.length} ${answerOf(shown.as).plural}`} ms={shown.took} steps={shown.steps} />
          <Toggle label="Debug" checked={debug} onChange={setDebug} />
        </div>
        {shown.missing.length > 0 && <p className="muted">No excerpt says: {shown.missing.join(', ')}</p>}
        {shown.uncovered.length > 0 && <p className="muted">Unanswered: {shown.uncovered.join(' · ')}</p>}
        {shown.holders.length > 0 && <p className="muted">Held by: {shown.holders.join(', ')}</p>}
        {debug ? (
          shown.body !== null && <pre className="md explore-raw">{JSON.stringify(shown.body, null, 2)}</pre>
        ) : shown.as === 'section' ? (
          <>
            <MapDocuments documents={shown.documents} />
            <SectionGrid sections={shown.sections} onOpen={setOpened} />
          </>
        ) : (
          <HitGrid
            results={shown.results}
            query={shown.asked[0] ?? ''}
            onOpen={setOpen}
            questions={shown.asked}
            reranked={shown.scoring.some((one) => one.step === 'rerank')}
          />
        )}
      </div>
      <SectionsModal section={opened} onClose={() => setOpened(null)} onOpen={setOpened} />
      <MatchModal match={open} query={shown.asked[0] ?? ''} scoring={shown.scoring} asked={{ questions: shown.asked, context: shown.context }} onClose={() => setOpen(null)} />
    </Shell>
  )
}
