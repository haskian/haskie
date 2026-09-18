import { useCallback, useEffect, useState } from 'react'
import { api, type Status } from './api'
import { Activity } from './components/Activity'
import { usePoll } from './hooks/usePoll'
import { Init } from './pages/Init'
import { Jobs } from './pages/Jobs'
import { Libraries } from './pages/Libraries'
import { Sessions } from './pages/Sessions'
import { Settings } from './pages/Settings'
import { Viewer } from './pages/Viewer'
import './App.css'

export type Route =
  | { name: 'libraries'; library?: string }
  | { name: 'viewer'; library: string; doc: string }
  | { name: 'sessions' }
  | { name: 'jobs' }
  | { name: 'settings' }

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [page, setPage] = useState<Route>({ name: 'libraries' })

  const refresh = useCallback(() => api.status().then(setStatus), [])
  useEffect(() => {
    refresh()
  }, [refresh])
  // poll while a model is downloading
  usePoll(status?.models.some((m) => m.state === 'loading' || m.state === 'pending') ?? false, refresh)

  if (!status) return <p className="muted">loading…</p>
  if (!status.initialized) return <Init onDone={refresh} />

  return (
    <div className="app">
      <nav>
        <strong>haskie</strong>
        <button onClick={() => setPage({ name: 'libraries' })}>Libraries</button>
        <button onClick={() => setPage({ name: 'sessions' })}>Sessions</button>
        <button onClick={() => setPage({ name: 'jobs' })}>Jobs</button>
        <button onClick={() => setPage({ name: 'settings' })}>Settings</button>
        <span className="muted">
          embedding: {status.embedding ? `${status.embedding.name} on ${status.device}` : 'none (full-text)'}
          {status.models.map((m) => (
            <span key={`${m.kind}:${m.name}`} className={m.state === 'error' ? 'error' : m.state === 'ready' ? '' : 'banner'} title={m.error ?? m.name}>
              {' '}· {m.kind} {m.state === 'ready' ? 'ready' : m.state === 'error' ? 'failed' : 'downloading…'}
            </span>
          ))}
        </span>
        <Activity onOpen={() => setPage({ name: 'jobs' })} />
      </nav>
      <main>
        {page.name === 'libraries' && <Libraries initial={page.library} navigate={setPage} />}
        {page.name === 'viewer' && <Viewer library={page.library} doc={page.doc} navigate={setPage} />}
        {page.name === 'sessions' && <Sessions />}
        {page.name === 'jobs' && <Jobs />}
        {page.name === 'settings' && <Settings />}
      </main>
    </div>
  )
}
