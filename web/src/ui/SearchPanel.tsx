import { useState } from 'react'
import type { Hit } from '../api'
import { HitGrid } from './HitGrid'
import { MatchModal } from './MatchModal'
import { SearchBox } from './SearchBox'
import { SearchTook } from './SearchTook'
import { errorText } from '../format'

/**
 * A search over one scope: the box, the hits it answered with, and the match one opens. The
 * caller decides which endpoint `run` asks.
 */
export function SearchPanel({ run, placeholder }: { run: (query: string) => Promise<Hit[]>; placeholder: string }) {
  const [query, setQuery] = useState('')
  const [asked, setAsked] = useState('') // the query the hits on screen answer
  const [hits, setHits] = useState<Hit[]>([])
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<Hit | null>(null)
  const [busy, setBusy] = useState(false)
  const [took, setTook] = useState<number | null>(null) // ms the last query took

  const submit = (): void => {
    const next = query.trim()
    setAsked(next)
    setError(null)
    if (next === '') {
      setHits([])
      setTook(null)
      return
    }
    const started = performance.now()
    setBusy(true)
    run(next)
      .then((found) => {
        setHits(found)
        setTook(Math.round(performance.now() - started))
      })
      .catch((cause: unknown) => setError(errorText(cause)))
      .finally(() => setBusy(false))
  }

  const clear = (): void => {
    setQuery('')
    setAsked('')
    setHits([])
    setError(null)
    setTook(null)
  }

  return (
    <>
      <SearchBox value={query} onChange={setQuery} placeholder={placeholder} onSubmit={submit} onClear={clear} busy={busy} />
      {error !== null && <p className="muted">{error}</p>}
      <SearchTook counts={`${hits.length} sections`} ms={took} />
      <HitGrid hits={hits} query={asked} onOpen={setOpen} />
      <MatchModal hit={open} query={asked} onClose={() => setOpen(null)} />
    </>
  )
}
