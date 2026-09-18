import { useCallback, useEffect, useState } from 'react'
import { api, type Activity as ActivityCounts } from '../api'
import { usePoll } from '../hooks/usePoll'

const BUSY_MS = 1500
const IDLE_MS = 5000

const line = (label: string, counts: { queued: number; running: number }) => `${label} ${counts.running} running, ${counts.queued} queued`

// Top-right indicator, on every view: jobs and tasks queued and running. Polls fast while
// anything is in flight and slowly otherwise, so new work started elsewhere still shows up.
export function Activity({ onOpen }: { onOpen: () => void }) {
  const [counts, setCounts] = useState<ActivityCounts | null>(null)
  const refresh = useCallback(() => api.activity().then(setCounts).catch(() => undefined), [])
  useEffect(() => {
    refresh()
  }, [refresh])
  const busy = counts !== null && Object.values(counts).some((c) => c.queued + c.running > 0)
  usePoll(true, refresh, busy ? BUSY_MS : IDLE_MS)

  if (!counts) return null
  return (
    <button className={`activity${busy ? ' busy' : ''}`} onClick={onOpen} title="open Jobs">
      {busy ? `${line('jobs', counts.jobs)} · ${line('tasks', counts.tasks)}` : 'idle'}
    </button>
  )
}
