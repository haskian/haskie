import { useEffect, useState } from 'react'
import { api, type EmbeddingProfile, type Options, type PipelineSettings, type SearchSettings } from '../api'
import { choices, docFor, EmbedderFacts, Field, Logo, Picker, profileOptions, SearchField } from '../ui'
import { errorText } from '../format'


// Shown instead of the shell until `~/.haskie` exists. The design has no init page, so this is
// composed from the tokens: the logo, a title, the pickers and one button.
export function Init({ onDone }: { onDone: () => void }) {
  const [options, setOptions] = useState<Options | null>(null)
  const [profile, setProfile] = useState<EmbeddingProfile>('granite-97m-multilingual')
  const [search, setSearch] = useState<SearchSettings | null>(null) // the server's defaults, then the picks
  const [descriptors, setDescriptors] = useState<PipelineSettings['descriptors'] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.options(), api.settings()])
      .then(([offered, defaults]) => {
        setOptions(offered)
        setSearch(defaults.search)
        setDescriptors(defaults.pipeline.descriptors)
      })
      .catch((cause: unknown) => setError(errorText(cause)))
  }, [])

  // Without an embedding model every mode is full-text, so there is no mode to choose.
  const embedded = profile !== 'none'

  const submit = async () => {
    if (search === null || descriptors === null) return
    setBusy(true)
    setError(null)
    try {
      await api.init({ profile, search: embedded ? search : { ...search, mode: 'fts' }, descriptors })
      onDone()
    } catch (cause) {
      setError(errorText(cause))
      setBusy(false)
    }
  }

  const profiles = profileOptions(options?.embedding_profiles ?? {}, options?.embedding_metadata ?? {})
  const doc = (key: string) => docFor(options?.docs ?? {}, key)

  return (
    <div className="page">
      <main className="body" style={{ gridTemplateColumns: '1fr', justifyItems: 'center' }}>
        <div className="sections" style={{ maxWidth: 520 }}>
          <Logo />
          <h1 className="title">
            Welcome to haskie
            <small className="muted">
              Pick the embedding model, how to search and who describes the sections. The model applies to every collection; changing it later means a full reindex.
              The search can change any time in the settings.
            </small>
          </h1>
          <Field label={doc('embedding').title}>
            <Picker options={profiles} value={profile} onChange={setProfile} ariaLabel="Embedding profile" />
            <EmbedderFacts model={options?.embedding_profiles[profile]} metadata={options?.embedding_metadata[profile]} />
          </Field>
          {options !== null && search !== null && descriptors !== null && (
            <>
              {embedded && <SearchField name="mode" search={search} options={options} onChange={setSearch} />}
              <SearchField name="reranker" search={search} options={options} onChange={setSearch} />
              {search.reranker === 'cross-encoder' && <SearchField name="reranker_model" search={search} options={options} onChange={setSearch} />}
              <Field label={doc('pipeline.descriptors').title} help={doc('pipeline.descriptors').description}>
                <Picker ariaLabel={doc('pipeline.descriptors').title} options={choices(options.descriptors)} value={descriptors} onChange={setDescriptors} />
              </Field>
            </>
          )}
          <button className="btn btn-primary" type="button" onClick={submit} disabled={busy || options === null}>
            {busy ? 'Initializing…' : 'Initialize ~/.haskie'}
          </button>
          {error !== null && <p className="muted">{error}</p>}
        </div>
      </main>
    </div>
  )
}
