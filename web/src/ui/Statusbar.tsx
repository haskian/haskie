import { Activity, Bot, Check, Clock, Settings, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api, type Activity as ActivityCounts, type ModelStatus, type Status } from '../api'
import { usePoll } from '../hooks/usePoll'
import { href, useRoute } from '../router'

const BUSY_MS = 1500
const IDLE_MS = 5000

// ponytail: the design's statusbar hints list the running and queued work by name.
// `/api/operations/activity` answers counts only, and the listing is a page away, so the hints
// are dropped rather than faked.
export function Statusbar({ status }: { status: Status }) {
  const route = useRoute()
  const [sessions, setSessions] = useState<number | null>(null)
  const [counts, setCounts] = useState<ActivityCounts | null>(null)

  // Sessions are written by agents, not by this app, so the count is re-read whenever the user
  // moves: a poll would ask for a number that changes once an hour.
  useEffect(() => {
    api
      .sessions()
      .then((all) => setSessions(Object.keys(all).length))
      .catch(() => undefined)
  }, [route.name])

  const refresh = useCallback(() => {
    api.activity().then(setCounts).catch(() => undefined)
  }, [])
  useEffect(refresh, [refresh])
  const busy = counts !== null && Object.values(counts).some((queue) => queue.queued + queue.running > 0)
  usePoll(true, refresh, busy ? BUSY_MS : IDLE_MS)

  const embedding = status.models.find((model) => model.kind === 'embedding') ?? null

  return (
    <div className="statusbar" role="status">
      <Bot className="icon" />
      <span className="statusbar-item">
        <span className="muted">Sessions</span>
        <span className="statusbar-counts">
          <b>{sessions ?? '—'}</b>
        </span>
      </span>
      <span className="statusbar-item">
        <span className="muted">Embedding</span>
        <span className="statusbar-counts">
          <EmbeddingState model={embedding} />
        </span>
      </span>
      <span className="spacer" />
      <a className="statusbar-item" href={href({ name: 'operations' })}>
        <Activity className="icon" />
        <Queue label="Operations" running={counts?.operations.running ?? 0} queued={counts?.operations.queued ?? 0} />
        <span className="muted">·</span>
        <Queue label="Tasks" running={counts?.tasks.running ?? 0} queued={counts?.tasks.queued ?? 0} />
      </a>
    </div>
  )
}

function EmbeddingState({ model }: { model: ModelStatus | null }) {
  if (model === null) return <b>full-text only</b>
  if (model.state === 'ready')
    return (
      <b className="done">
        <Check className="icon" />
        {model.name} · ready
      </b>
    )
  if (model.state === 'error')
    return (
      <b>
        <X className="icon" />
        {model.name} · error
      </b>
    )
  return (
    <b className="running">
      <Settings className="icon spin" />
      {model.name} · {model.state}
    </b>
  )
}

function Queue({ label, running, queued }: { label: string; running: number; queued: number }) {
  return (
    <span className="statusbar-item">
      <span className="muted">{label}</span>
      <span className="statusbar-counts">
        <b className="running">
          <Settings className="icon spin" />
          {running}
        </b>
        <b className="queued">
          <Clock className="icon" />
          {queued}
        </b>
      </span>
    </span>
  )
}
