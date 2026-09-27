import { useState } from 'react'
import type { ScoreStep, StepTiming, Timed } from '../api'
import { HitGrid } from './HitGrid'
import type { Match } from './match'
import { MatchModal } from './MatchModal'
import { SearchBox } from './SearchBox'
import { SearchTook } from './SearchTook'
import { errorText } from '../format'

/**
 * A search over one scope: the box, the results it answered with, and the match one opens, with
 * the steps the search took and how it scored. The caller decides which search `run` asks, and
 * `plural` names what it answers with.
 */
export function SearchPanel({ run, placeholder, plural }: { run: (query: string) => Promise<Timed<Match[]>>; placeholder: string; plural: string }) {
  const [query, setQuery] = useState('')
  const [asked, setAsked] = useState('') // the query the results on screen answer
  const [results, setResults] = useState<Match[]>([])
  const [steps, setSteps] = useState<StepTiming[]>([]) // how long each step of it took
  const [scoring, setScoring] = useState<ScoreStep[]>([]) // and how its scores came to be
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [busy, setBusy] = useState(false)
  const [took, setTook] = useState<number | null>(null) // ms the last query took

  const clear = (): void => {
    setQuery('')
    setAsked('')
    setResults([])
    setSteps([])
    setScoring([])
    setError(null)
    setTook(null)
  }

  const submit = (): void => {
    const next = query.trim()
    if (next === '') {
      clear()
      return
    }
    setAsked(next)
    setError(null)
    const started = performance.now()
    setBusy(true)
    run(next)
      .then((found) => {
        setResults(found.body)
        setSteps(found.steps)
        setScoring(found.scoring)
        setTook(Math.round(performance.now() - started))
      })
      .catch((cause: unknown) => setError(errorText(cause)))
      .finally(() => setBusy(false))
  }

  return (
    <>
      <SearchBox value={query} onChange={setQuery} placeholder={placeholder} onSubmit={submit} onClear={clear} busy={busy} />
      {error !== null && <p className="muted">{error}</p>}
      <SearchTook counts={`${results.length} ${plural}`} ms={took} steps={steps} />
      <HitGrid results={results} query={asked} onOpen={setOpen} />
      <MatchModal match={open} query={asked} scoring={scoring} onClose={() => setOpen(null)} />
    </>
  )
}
