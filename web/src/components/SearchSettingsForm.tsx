import type { Fusion, Options, Reranker, SearchMode, SearchOverrides, SearchSettings } from '../api'
import { Field } from './Field'

type Props = {
  value: SearchSettings | SearchOverrides
  defaults?: SearchSettings // when set, fields are overrides and may be null (= default)
  options: Options
  onChange: (next: SearchSettings | SearchOverrides) => void
}

// One form for both user-level (concrete values) and library-level (nullable overrides) settings.
export function SearchSettingsForm({ value, defaults, options, onChange }: Props) {
  const doc = (k: keyof SearchSettings) => options.docs[`search.${k}`]
  const set = <K extends keyof SearchSettings>(k: K, v: SearchSettings[K] | null) => onChange({ ...value, [k]: v })
  const num = (k: keyof SearchSettings, step = 1, min = 0) => (
    <Field key={k} name={k} doc={doc(k)}>
      <input
        type="number"
        step={step}
        min={min}
        value={value[k] ?? ''}
        placeholder={defaults ? `default ${defaults[k]}` : undefined}
        onChange={(e) => set(k, e.target.value === '' ? (defaults ? null : 0) : Number(e.target.value))}
      />
    </Field>
  )
  const choice = <K extends 'mode' | 'fusion' | 'reranker' | 'reranker_model'>(k: K, items: readonly string[]) => (
    <Field key={k} name={k} doc={doc(k)}>
      <select value={value[k] ?? ''} onChange={(e) => set(k, (e.target.value || null) as SearchSettings[K] | null)}>
        {defaults && <option value="">default ({defaults[k]})</option>}
        {items.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
    </Field>
  )
  const eff = <K extends keyof SearchSettings>(k: K) => value[k] ?? defaults?.[k]
  const hybrid = eff('mode') === 'hybrid'
  const rerank = eff('reranker') === 'cross-encoder'
  return (
    <>
      {num('limit', 1, 1)}
      {choice('mode', options.search_modes as SearchMode[])}
      {hybrid && choice('fusion', options.fusions as Fusion[])}
      {hybrid && eff('fusion') === 'rrf' && num('rrf_k', 1, 1)}
      {hybrid && eff('fusion') === 'linear' && num('vector_weight', 0.05)}
      {hybrid && eff('fusion') === 'linear' && num('bm25_weight', 0.05)}
      {eff('mode') !== 'fts' && num('nprobes', 1, 1)}
      {eff('mode') !== 'fts' && num('refine_factor', 1, 1)}
      {choice('reranker', options.rerankers as Reranker[])}
      {rerank && choice('reranker_model', options.reranker_models)}
      {(hybrid || rerank) && num('candidates', 1, 1)}
    </>
  )
}
