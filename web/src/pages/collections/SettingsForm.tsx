import { useState, type ReactNode } from 'react'
import type { Chunker, ChunkSettings, CollectionOverrides, Fusion, Options, Reranker, ScoreFold, SearchMode, SearchSettings } from '../../api'
import { effectiveSearch, Field, Picker, rerankerOption, SEARCH_BOUNDS, visibleSearchFields, type NumericKeys, type PickerOption } from '../../ui'

type SearchField = keyof SearchSettings

type NumberField = NumericKeys<SearchSettings>

interface NumberSpec {
  step: number
  min: number
  max?: number
}

const DEFAULT_OPTION_LABEL = 'Default'
// A switch a collection may override, or leave to the user settings: the picker's first option.
const ON = 'on'
const OFF = 'off'
const onOff = (value: boolean): string => (value ? ON : OFF)

/**
 * A collection's chunking and search overrides. Every field may be left empty, and an empty
 * field means "whatever the user settings say": the picker's first option and the number
 * placeholders show what that is right now.
 *
 * The draft is seeded from `overrides` once and then left alone, so a poll landing behind the
 * form cannot overwrite what is being typed.
 */
export function SettingsForm({
  overrides,
  effective,
  searchDefaults,
  options,
  outdated,
  busy,
  onSave,
  children,
}: {
  overrides: CollectionOverrides
  effective: ChunkSettings // what the chunking overrides resolve to
  searchDefaults: SearchSettings // what the search overrides resolve to
  options: Options
  outdated: boolean
  busy: boolean
  onSave: (next: CollectionOverrides) => void
  children?: ReactNode // the actions that are not "Save": index, delete, and how they are going
}) {
  const [draft, setDraft] = useState<CollectionOverrides>(overrides)
  const current = effectiveSearch(draft.search, searchDefaults)

  const label = (key: string, fallback: string): string => options.docs[key]?.title ?? fallback
  const help = (key: string): string | undefined => options.docs[key]?.description

  const setSearch = <K extends SearchField>(key: K, next: SearchSettings[K] | null): void =>
    setDraft({ ...draft, search: { ...draft.search, [key]: next } })

  const numberInput = (
    key: string,
    docKey: string,
    value: number | null,
    placeholder: number | string, // the default the empty field falls back to
    spec: NumberSpec,
    onChange: (next: number | null) => void,
  ): ReactNode => (
    <Field key={key} label={label(docKey, key)} help={help(docKey)}>
      <input
        className="input"
        type="number"
        step={spec.step}
        min={spec.min}
        max={spec.max}
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
    toOption: (item: string) => PickerOption<string> = (item) => ({ value: item, label: item }),
  ): ReactNode => {
    const choices: PickerOption<string>[] = [{ value: '', label: DEFAULT_OPTION_LABEL, sub: String(fallback) }, ...items.map(toOption)]
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
      (item) => rerankerOption(item, options.reranker_metadata),
    ),
    rerank_with_context: enumInput(
      'rerank_with_context',
      'search.rerank_with_context',
      [ON, OFF],
      draft.search.rerank_with_context == null ? null : onOff(draft.search.rerank_with_context),
      onOff(searchDefaults.rerank_with_context),
      (next) => setSearch('rerank_with_context', next === null ? null : next === ON),
    ),
    min_rerank_score: numberInput(
      'min_rerank_score',
      'search.min_rerank_score',
      draft.search.min_rerank_score ?? null,
      searchDefaults.min_rerank_score ?? "the reranker's floor",
      { min: 0, max: 1, step: 0.01 },
      (next) => setSearch('min_rerank_score', next),
    ),
    score_fold: enumInput('score_fold', 'search.score_fold', options.score_folds, draft.search.score_fold, searchDefaults.score_fold, (next) =>
      setSearch('score_fold', next as ScoreFold | null),
    ),
    candidates: searchNumber('candidates'),
    min_passage_chars: searchNumber('min_passage_chars'),
    max_passage_grow: searchNumber('max_passage_grow'),
    max_section_chars: searchNumber('max_section_chars'),
    max_answer_chars: searchNumber('max_answer_chars'),
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
        {numberInput('chunk_merge_below', 'conversion.chunk_merge_below', draft.chunk_merge_below, effective.chunk_merge_below, { step: 1, min: 0, max: 100 }, (next) =>
          setDraft({ ...draft, chunk_merge_below: next }),
        )}
        {enumInput(
          'chunk_frame',
          'conversion.chunk_frame',
          [ON, OFF],
          draft.chunk_frame === null || draft.chunk_frame === undefined ? null : onOff(draft.chunk_frame),
          onOff(effective.chunk_frame),
          (next) => setDraft({ ...draft, chunk_frame: next === null ? null : next === ON }),
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
