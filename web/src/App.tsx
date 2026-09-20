import { useCallback, useEffect, useState } from 'react'
import { api, type Status } from './api'
import { Activity } from './components/Activity'
import { usePoll } from './hooks/usePoll'
import { Collections } from './pages/Collections'
import { Documents } from './pages/Documents'
import { Init } from './pages/Init'
import { Jobs } from './pages/Jobs'
import { Sessions } from './pages/Sessions'
import { Settings } from './pages/Settings'
import { Viewer } from './pages/Viewer'

// A document is imported once and addressed by name alone, so the viewer needs no collection.
export type Route =
  | { name: 'documents' }
  | { name: 'collections' }
  | { name: 'viewer'; doc: string }
  | { name: 'sessions' }
  | { name: 'jobs' }
  | { name: 'settings' }

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [page, setPage] = useState<Route>({ name: 'documents' })

  const refresh = useCallback(() => api.status().then(setStatus), [])
  useEffect(() => {
    void api.options() // started here so it travels with the status request, not after it
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
        <button onClick={() => setPage({ name: 'documents' })}>Documents</button>
        <button onClick={() => setPage({ name: 'collections' })}>Collections</button>
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
        {page.name === 'documents' && <Documents navigate={setPage} />}
        {page.name === 'collections' && <Collections navigate={setPage} />}
        {page.name === 'viewer' && <Viewer doc={page.doc} navigate={setPage} />}
        {page.name === 'sessions' && <Sessions />}
        {page.name === 'jobs' && <Jobs />}
        {page.name === 'settings' && <Settings />}
      </main>
    </div>
  )
}
