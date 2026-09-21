import { useCallback, useState } from 'react'
import { errorText } from '../format'

export interface Run {
  run: (fn: () => Promise<unknown>) => Promise<void>
  busy: boolean
  error: string | null
  setError: (message: string | null) => void
}

// One mutation, done the way every view does it: clear the last error, run the request, re-read
// what is on screen, and keep a failure on the page instead of throwing it away. `busy` is raised
// for the whole round trip so a form can lock itself while it runs.
export function useRun(refresh: () => Promise<unknown>): Run {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = useCallback(
    async (fn: () => Promise<unknown>) => {
      setError(null)
      setBusy(true)
      try {
        await fn()
        await refresh()
      } catch (e) {
        setError(errorText(e))
      } finally {
        setBusy(false)
      }
    },
    [refresh],
  )

  return { run, busy, error, setError }
}
