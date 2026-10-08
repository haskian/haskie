import { Activity, Bot, Check, Clock, LoaderCircle, Settings, X } from 'lucide-react'
import { Fragment, useCallback, useEffect, useState, type ReactNode } from 'react'
import { api, type Activity as ActivityCounts, type ModelStatus, type Status } from '../api'
import { usePoll } from '../hooks/usePoll'
import { onWorkStarted } from '../hooks/workStarted'
import { href, useRoute } from '../router'

const BUSY_MS = 1500
const IDLE_MS = 5000

const isBusy = (counts: ActivityCounts) =>
  Object.values(counts).some((queue) => queue.queued + queue.running > 0)

// The design's statusbar hints list the running and queued work by name.
// `/api/operations/activity` answers counts only, and the listing is a page away, so the hints
// are dropped rather than faked.
export function Statusbar({ status, refreshStatus }: { status: Status; refreshStatus?: () => void }) {
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

  // search: the embedding model and a reranker when one is on, loaded for the life of the server;
  // knowledge: the models indexing asks (the describer, the vocabulary's embedder), loaded on
  // first use and freed once idle, and only there when the settings ask a language model
  const search = status.models.filter((model) => model.group === 'search')
  const knowledge = status.models.filter((model) => model.group === 'knowledge')

  // A knowledge model changes state with the work that loads it, and is freed once idle: the
  // models are read again with the queues while work runs, or while one is in memory or on its way.
  const knowledgeMoving = knowledge.some((model) => model.state !== 'downloaded' && model.state !== 'error')
  const watchKnowledge = knowledge.length > 0
  const refresh = useCallback(() => {
    api
      .activity()
      .then((fresh) => {
        setCounts(fresh)
        if (watchKnowledge && (knowledgeMoving || isBusy(fresh))) refreshStatus?.()
      })
      .catch(() => undefined)
  }, [watchKnowledge, knowledgeMoving, refreshStatus])
  useEffect(refresh, [refresh])
  useEffect(() => onWorkStarted(refresh), [refresh])
  usePoll(true, refresh, counts !== null && isBusy(counts) ? BUSY_MS : IDLE_MS)

  return (
    <div className="statusbar" role="status">
      <Bot className="icon" />
      <span className="statusbar-item">
        <span className="muted">Sessions</span>
        <span className="statusbar-counts">
          <b>{sessions ?? '—'}</b>
        </span>
      </span>
      <Models label="Search" models={search} none="full-text only" />
      {knowledge.length > 0 && <Models label="Knowledge" models={knowledge} />}
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

// Each state a model can be in, least advanced first, the order a group lists them in: what the
// hint says after the model's name, and its icon. In memory, a turning green gear; on disk and
// loaded when used, a white check; on its way, a grey spinner; failed, a cross.
const STATES: [ModelStatus['state'], { words: string; className?: string; icon: ReactNode }][] = [
  ['error', { words: 'failed', icon: <X className="icon" /> }],
  ['pending', { words: 'waiting', className: 'queued', icon: <LoaderCircle className="icon spin spin-fast" /> }],
  [
    'loading',
    { words: 'downloading or loading', className: 'queued', icon: <LoaderCircle className="icon spin spin-fast" /> },
  ],
  ['downloaded', { words: 'downloaded, loaded when used', className: 'idle', icon: <Check className="icon" /> }],
  ['ready', { words: 'loaded', className: 'running', icon: <Settings className="icon spin" /> }],
]
const WORDS = Object.fromEntries(STATES.map(([state, { words }]) => [state, words])) as Record<
  ModelStatus['state'],
  string
>

/** One group of models: an icon per state its models are in, each with how many are in it, the
 *  least advanced first (one loaded and one downloading reads "1 downloading / 1 loaded"). The
 *  hint, which opens downwards, names each model with its own state, kind and error. */
function Models({ label, models, none }: { label: string; models: ModelStatus[]; none?: string }) {
  const present = STATES.map(([state, look]) => ({
    state,
    look,
    count: models.filter((model) => model.state === state).length,
  })).filter(({ count }) => count > 0)
  return (
    <span className="statusbar-item" tabIndex={0}>
      <span className="muted">{label}</span>
      <span className="statusbar-counts">
        {present.length === 0 ? (
          <b>{none}</b>
        ) : (
          present.map(({ state, look, count }) => (
            <b key={state} className={look.className} aria-label={state === 'ready' ? 'loaded' : state}>
              {look.icon}
              {count}
            </b>
          ))
        )}
      </span>
      {models.length > 0 && (
        <span className="hint" role="tooltip">
          <span className="hint-rows">
            {models.map((model) => (
              <Fragment key={model.name}>
                <span>{model.name}</span>
                <span className="muted">{model.error ?? model.kind}</span>
                <span className="code">{WORDS[model.state]}</span>
              </Fragment>
            ))}
          </span>
        </span>
      )}
    </span>
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
