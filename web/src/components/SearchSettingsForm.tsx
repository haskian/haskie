import type { Options, SearchOverrides, SearchSettings } from '../api'
import { Field, NumberField } from './Field'

type NumberKey = 'limit' | 'candidates' | 'rrf_k' | 'vector_weight' | 'bm25_weight' | 'nprobes' | 'refine_factor'
type ChoiceKey = 'mode' | 'fusion' | 'reranker' | 'reranker_model'

type Props<V extends SearchOverrides> = {
  value: V
  defaults?: SearchSettings // when set, fields are overrides and may be null (= default)
  options: Options
  onChange: (next: V) => void
}

// One form for both user-level (concrete values) and collection-level (nullable overrides) settings.
export function SearchSettingsForm<V extends SearchOverrides>({ value, defaults, options, onChange }: Props<V>) {
  const doc = (k: keyof SearchSettings) => options.docs[`search.${k}`]
  const set = <K extends keyof SearchSettings>(k: K, v: SearchSettings[K] | null) => onChange({ ...value, [k]: v })
  const num = (k: NumberKey, step = 1, min = 0) => (
    <NumberField
      key={k}
      doc={doc(k)}
      value={value[k]}
      min={min}
      step={step}
      placeholder={defaults ? `default ${defaults[k]}` : undefined}
      onChange={(next) => set(k, next ?? (defaults ? null : 0))}
    />
  )
  const choice = <K extends ChoiceKey>(k: K, items: readonly string[]) => (
    <Field key={k} doc={doc(k)}>
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
      {choice('mode', options.search_modes)}
      {hybrid && choice('fusion', options.fusions)}
      {hybrid && eff('fusion') === 'rrf' && num('rrf_k', 1, 1)}
      {hybrid && eff('fusion') === 'linear' && num('vector_weight', 0.05)}
      {hybrid && eff('fusion') === 'linear' && num('bm25_weight', 0.05)}
      {eff('mode') !== 'fts' && num('nprobes', 1, 1)}
      {eff('mode') !== 'fts' && num('refine_factor', 1, 1)}
      {choice('reranker', options.rerankers)}
      {rerank && choice('reranker_model', options.reranker_models)}
      {(hybrid || rerank) && num('candidates', 1, 1)}
    </>
  )
}
