import { useState } from 'react'
import type { Timed } from '../api'
import { HitGrid } from './HitGrid'
import type { Match } from './match'
import { MatchModal } from './MatchModal'
import { ModalStatus } from './ModalStatus'
import { SearchBox } from './SearchBox'
import { SearchTook } from './SearchTook'
import { errorText } from '../format'

/** The search whose results are on screen: what it answered, the query it answers, and the ms it
 *  took, null before the first search so the line above the results stays blank. */
interface Shown extends Timed<Match[]> {
  asked: string
  took: number | null
}

const NOTHING: Shown = { body: [], steps: [], scoring: [], asked: '', took: null }

/**
 * A search over one scope: the box, the results it answered with, and the match one opens, with
 * the steps the search took and how it scored. The caller decides which search `run` asks, and
 * `plural` names what it answers with.
 */
export function SearchPanel({ run, placeholder, plural, active = true }: { active?: boolean; run: (query: string) => Promise<Timed<Match[]>>; placeholder: string; plural: string }) {
  const [query, setQuery] = useState('')
  const [shown, setShown] = useState<Shown>(NOTHING)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Match | null>(null)
  const [busy, setBusy] = useState(false)

  const clear = (): void => {
    setQuery('')
    setShown(NOTHING)
    setError(null)
  }

  const submit = (): void => {
    const asked = query.trim()
    if (asked === '') {
      clear()
      return
    }
    setError(null)
    const started = performance.now()
    setBusy(true)
    run(asked)
      .then((found) => setShown({ ...found, asked, took: Math.round(performance.now() - started) }))
      .catch((cause: unknown) => setError(errorText(cause)))
      .finally(() => setBusy(false))
  }

  return (
    <>
      <SearchBox value={query} onChange={setQuery} placeholder={placeholder} onSubmit={submit} onClear={clear} busy={busy} />
      {active && error !== null && <ModalStatus tone="error">{error}</ModalStatus>}
      <SearchTook counts={`${shown.body.length} ${plural}`} ms={shown.took} steps={shown.steps} />
      <HitGrid results={shown.body} query={shown.asked} onOpen={setOpen} />
      <MatchModal match={open} query={shown.asked} scoring={shown.scoring} onClose={() => setOpen(null)} />
    </>
  )
}
