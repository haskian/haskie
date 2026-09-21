import { useState, type ReactNode } from 'react'
import type { Chunker, ChunkSettings, CollectionSettings, Fusion, Options, Reranker, SearchMode, SearchSettings } from '../../api'
import { effectiveSearch, Field, Picker, SEARCH_BOUNDS, visibleSearchFields, type NumericKeys, type PickerOption } from '../../ui'

type SearchField = keyof SearchSettings

type NumberField = NumericKeys<SearchSettings>

interface NumberSpec {
  step: number
  min: number
}

const DEFAULT_OPTION_LABEL = 'Default'

/**
 * A collection's chunking and search overrides. Every field may be left empty, and an empty
 * field means "whatever the user settings say": the picker's first option and the number
 * placeholders show what that is right now.
 *
 * The draft is seeded from `settings` once and then left alone, so a poll landing behind the
 * form cannot overwrite what is being typed.
 */
export function SettingsForm({
  settings,
  effective,
  searchDefaults,
  options,
  outdated,
  busy,
  onSave,
  children,
}: {
  settings: CollectionSettings
  effective: ChunkSettings // what the chunking overrides resolve to
  searchDefaults: SearchSettings // what the search overrides resolve to
  options: Options
  outdated: boolean
  busy: boolean
  onSave: (next: CollectionSettings) => void
  children?: ReactNode // the actions that are not "Save": index, delete, and how they are going
}) {
  const [draft, setDraft] = useState<CollectionSettings>(settings)
  const current = effectiveSearch(draft.search, searchDefaults)

  const label = (key: string, fallback: string): string => options.docs[key]?.title ?? fallback
  const help = (key: string): string | undefined => options.docs[key]?.description

  const setSearch = <K extends SearchField>(key: K, next: SearchSettings[K] | null): void =>
    setDraft({ ...draft, search: { ...draft.search, [key]: next } })

  const numberInput = (
    key: string,
    docKey: string,
    value: number | null,
    placeholder: number,
    spec: NumberSpec,
    onChange: (next: number | null) => void,
  ): ReactNode => (
    <Field key={key} label={label(docKey, key)} help={help(docKey)}>
      <input
        className="input"
        type="number"
        step={spec.step}
        min={spec.min}
        value={value ?? ''}
        placeholder={`default ${placeholder}`}
        onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))}
      />
    </Field>
  )

  // The first option is the default, and picking it clears the override.
  const enumInput = (
    key: string,
    docKey: string,
    items: readonly string[],
    value: string | null,
    fallback: string,
    onChange: (next: string | null) => void,
  ): ReactNode => {
    const choices: PickerOption<string>[] = [
      { value: '', label: DEFAULT_OPTION_LABEL, sub: String(fallback) },
      ...items.map((item) => ({ value: item, label: item })),
    ]
    return (
      <Field key={key} label={label(docKey, key)} help={help(docKey)}>
        <Picker options={choices} value={value ?? ''} onChange={(next) => onChange(next === '' ? null : next)} ariaLabel={key} />
      </Field>
    )
  }

  const searchNumber = (key: NumberField): ReactNode =>
    numberInput(key, `search.${key}`, draft.search[key], searchDefaults[key], SEARCH_BOUNDS[key], (next) => setSearch(key, next))

  const searchFields: Record<SearchField, ReactNode> = {
    limit: searchNumber('limit'),
    mode: enumInput('mode', 'search.mode', options.search_modes, draft.search.mode, searchDefaults.mode, (next) =>
      setSearch('mode', next as SearchMode | null),
    ),
    fusion: enumInput('fusion', 'search.fusion', options.fusions, draft.search.fusion, searchDefaults.fusion, (next) =>
      setSearch('fusion', next as Fusion | null),
    ),
    rrf_k: searchNumber('rrf_k'),
    vector_weight: searchNumber('vector_weight'),
    bm25_weight: searchNumber('bm25_weight'),
    nprobes: searchNumber('nprobes'),
    refine_factor: searchNumber('refine_factor'),
    reranker: enumInput('reranker', 'search.reranker', options.rerankers, draft.search.reranker, searchDefaults.reranker, (next) =>
      setSearch('reranker', next as Reranker | null),
    ),
    reranker_model: enumInput(
      'reranker_model',
      'search.reranker_model',
      options.reranker_models,
      draft.search.reranker_model,
      searchDefaults.reranker_model,
      (next) => setSearch('reranker_model', next),
    ),
    candidates: searchNumber('candidates'),
  }

  return (
    <div className="collection-settings">
      {outdated && <p className="muted">index was built by an older version — use Index all to rebuild it</p>}
      <span className="mono muted">Chunking</span>
      <div className="collection-fields">
        {enumInput('chunker', 'conversion.chunker', options.chunkers, draft.chunker, effective.chunker, (next) =>
          setDraft({ ...draft, chunker: next as Chunker | null }),
        )}
        {numberInput('chunk_size', 'conversion.chunk_size', draft.chunk_size, effective.chunk_size, { step: 1, min: 1 }, (next) =>
          setDraft({ ...draft, chunk_size: next }),
        )}
        {numberInput('chunk_overlap', 'conversion.chunk_overlap', draft.chunk_overlap, effective.chunk_overlap, { step: 1, min: 0 }, (next) =>
          setDraft({ ...draft, chunk_overlap: next }),
        )}
      </div>
      <span className="mono muted">Search</span>
      <div className="collection-fields">{visibleSearchFields(current).map((key) => searchFields[key])}</div>
      <div className="row row-loose">
        <button className="btn btn-primary" type="button" disabled={busy} onClick={() => onSave(draft)}>
          Save
        </button>
        {children}
      </div>
    </div>
  )
}
