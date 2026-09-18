import { useEffect, useState } from 'react'
import { api, type EmbeddingProfile, type Options } from '../api'

const HINTS: Record<EmbeddingProfile, string> = {
  none: 'Full-text search only. No model download.',
  compact: 'Small English model (~130 MB). Fast, good default.',
  quality: 'Large English model (~1.3 GB). Better recall, slower.',
  multilingual: 'Large multilingual model (~2.2 GB).',
}

export function Init({ onDone }: { onDone: () => void }) {
  const [options, setOptions] = useState<Options | null>(null)
  const [profile, setProfile] = useState<EmbeddingProfile>('compact')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.options().then(setOptions)
  }, [])

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.init(profile)
      onDone()
    } catch (e) {
      setError(String(e))
      setBusy(false)
    }
  }

  return (
    <div className="init">
      <h1>Welcome to haskie</h1>
      <p>Pick the embedding model. It applies to every library; changing it later means a full reindex.</p>
      {options &&
        (Object.keys(options.embedding_profiles) as EmbeddingProfile[]).map((p) => (
          <label key={p} className="radio">
            <input type="radio" checked={profile === p} onChange={() => setProfile(p)} disabled={busy} />
            <span>
              <strong>{p}</strong> {options.embedding_profiles[p]?.name ?? ''}
              <br />
              <small className="muted">{HINTS[p]}</small>
            </span>
          </label>
        ))}
      <button onClick={submit} disabled={busy || !options}>
        {busy ? 'initializing…' : 'Initialize ~/.haskie'}
      </button>
      {error && <p className="error">{error}</p>}
    </div>
  )
}
