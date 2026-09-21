import { useEffect, useState } from 'react'
import { api, type EmbeddingProfile, type Options } from '../api'
import { Logo, Picker, type PickerOption } from '../ui'
import { errorText } from '../format'

const HINTS: Record<EmbeddingProfile, string> = {
  none: 'Full-text search only. No model download.',
  compact: 'Small English model (~130 MB). Fast, good default.',
  quality: 'Large English model (~1.3 GB). Better recall, slower.',
  multilingual: 'Large multilingual model (~2.2 GB).',
}

// Shown instead of the shell until `~/.haskie` exists. The design has no init page, so this is
// composed from the tokens: the logo, a title, one picker and one button.
export function Init({ onDone }: { onDone: () => void }) {
  const [options, setOptions] = useState<Options | null>(null)
  const [profile, setProfile] = useState<EmbeddingProfile>('compact')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.options().then(setOptions).catch((cause: unknown) => setError(errorText(cause)))
  }, [])

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.init(profile)
      onDone()
    } catch (cause) {
      setError(errorText(cause))
      setBusy(false)
    }
  }

  const profiles: PickerOption<EmbeddingProfile>[] = Object.entries(options?.embedding_profiles ?? {}).map(([value, model]) => {
    const key = value as EmbeddingProfile
    return { value: key, label: key, sub: model === null ? HINTS[key] : `${model.name} · ${HINTS[key]}` }
  })

  return (
    <div className="page">
      <main className="body" style={{ gridTemplateColumns: '1fr', justifyItems: 'center' }}>
        <div className="sections" style={{ maxWidth: 520 }}>
          <Logo />
          <h1 className="title">
            Welcome to haskie
            <small className="muted">
              Pick the embedding model. It applies to every collection; changing it later means a full reindex.
            </small>
          </h1>
          <Picker options={profiles} value={profile} onChange={setProfile} ariaLabel="Embedding profile" />
          <button className="btn btn-primary" type="button" onClick={submit} disabled={busy || options === null}>
            {busy ? 'Initializing…' : 'Initialize ~/.haskie'}
          </button>
          {error !== null && <p className="muted">{error}</p>}
        </div>
      </main>
    </div>
  )
}
