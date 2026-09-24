import { Activity, Bot, Check, Clock, Settings, X } from 'lucide-react'
import { Fragment, useCallback, useEffect, useState } from 'react'
import { api, type Activity as ActivityCounts, type ModelStatus, type Status } from '../api'
import { usePoll } from '../hooks/usePoll'
import { href, useRoute } from '../router'

const BUSY_MS = 1500
const IDLE_MS = 5000

// the design's statusbar hints list the running and queued work by name.
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

  const embedding = status.models.filter((model) => model.kind === 'embedding')
  // a reranker is in the list only when one is on, in the settings or a collection's overrides
  const rerankers = status.models.filter((model) => model.kind === 'reranker')

  return (
    <div className="statusbar" role="status">
      <Bot className="icon" />
      <span className="statusbar-item">
        <span className="muted">Sessions</span>
        <span className="statusbar-counts">
          <b>{sessions ?? '—'}</b>
        </span>
      </span>
      <Models label="Embedding" models={embedding} none="full-text only" />
      {rerankers.length > 0 && <Models label="Reranker" models={rerankers} />}
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

// The worst state of a kind's models is the one its icon shows: one reranker still loading
// means a search that asks for it waits.
const WORST_FIRST: ModelStatus['state'][] = ['error', 'loading', 'pending', 'ready']

/** One kind of model as an icon: a check once ready, a spinner while it loads, a cross on an
 *  error. The names and each one's state are in the hint, which opens downwards. */
function Models({ label, models, none }: { label: string; models: ModelStatus[]; none?: string }) {
  const state = WORST_FIRST.find((one) => models.some((model) => model.state === one))
  return (
    <span className="statusbar-item" tabIndex={0}>
      <span className="muted">{label}</span>
      <span className="statusbar-counts">
        {state === undefined ? <b>{none}</b> : <StateIcon state={state} />}
      </span>
      {models.length > 0 && (
        <span className="hint" role="tooltip">
          <span className="hint-rows">
            {models.map((model) => (
              <Fragment key={model.name}>
                <span>{model.name}</span>
                <span className="muted">{model.error ?? ''}</span>
                <span className="code">{model.state}</span>
              </Fragment>
            ))}
          </span>
        </span>
      )}
    </span>
  )
}

function StateIcon({ state }: { state: ModelStatus['state'] }) {
  if (state === 'ready')
    return (
      <b className="done" aria-label="ready">
        <Check className="icon" />
      </b>
    )
  if (state === 'error')
    return (
      <b aria-label="error">
        <X className="icon" />
      </b>
    )
  return (
    <b className="running" aria-label={state}>
      <Settings className="icon spin" />
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
