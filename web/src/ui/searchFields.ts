import type { SearchOverrides, SearchSettings } from '../api'

/** The keys of `T` that hold a `V`, so a table of number fields cannot name a string one. */
export type KeysOf<T, V> = { [K in keyof T]: T[K] extends V ? K : never }[keyof T]

export type NumericKeys<T> = KeysOf<T, number>

/**
 * What a search number input accepts. One table, so the user settings page and a collection's
 * override form cannot disagree about which values are legal.
 */
export const SEARCH_BOUNDS: Record<NumericKeys<SearchSettings>, { min: number; max?: number; step: number }> = {
  limit: { min: 1, step: 1 },
  candidates: { min: 1, step: 1 },
  rrf_k: { min: 1, step: 1 },
  vector_weight: { min: 0, step: 0.05 },
  bm25_weight: { min: 0, step: 0.05 },
  nprobes: { min: 1, step: 1 },
  refine_factor: { min: 1, step: 1 },
  min_passage_chars: { min: 0, step: 50 },
  max_passage_grow: { min: 0, step: 1 },
  grow_bias: { min: -1, max: 1, step: 0.05 },
  max_section_chars: { min: 1, step: 500 },
  max_answer_chars: { min: 1, step: 1000 },
}

/**
 * What a collection searches with: its own value where it overrides one, the default
 * where it does not. `0` is a value, so only `null` falls back.
 */
export function effectiveSearch(overrides: SearchOverrides, defaults: SearchSettings): SearchSettings {
  return {
    limit: overrides.limit ?? defaults.limit,
    candidates: overrides.candidates ?? defaults.candidates,
    mode: overrides.mode ?? defaults.mode,
    fusion: overrides.fusion ?? defaults.fusion,
    rrf_k: overrides.rrf_k ?? defaults.rrf_k,
    vector_weight: overrides.vector_weight ?? defaults.vector_weight,
    bm25_weight: overrides.bm25_weight ?? defaults.bm25_weight,
    nprobes: overrides.nprobes ?? defaults.nprobes,
    refine_factor: overrides.refine_factor ?? defaults.refine_factor,
    reranker: overrides.reranker ?? defaults.reranker,
    reranker_model: overrides.reranker_model ?? defaults.reranker_model,
    rerank_with_context: overrides.rerank_with_context ?? defaults.rerank_with_context,
    min_rerank_score: overrides.min_rerank_score ?? defaults.min_rerank_score,
    score_fold: overrides.score_fold ?? defaults.score_fold,
    rerank_excerpts: overrides.rerank_excerpts ?? defaults.rerank_excerpts,
    fill_values: overrides.fill_values ?? defaults.fill_values,
    min_passage_chars: overrides.min_passage_chars ?? defaults.min_passage_chars,
    max_passage_grow: overrides.max_passage_grow ?? defaults.max_passage_grow,
    grow_bias: overrides.grow_bias ?? defaults.grow_bias,
    max_section_chars: overrides.max_section_chars ?? defaults.max_section_chars,
    max_answer_chars: overrides.max_answer_chars ?? defaults.max_answer_chars,
  }
}

/**
 * Which search fields a form shows, in order: a field is only asked for when the effective
 * settings make it do something. Fusion weights belong to a hybrid query, and probes to a
 * vector one. The reranker model, and whether one reads the shared context, belong to a reranker.
 * The candidate pool belongs to whichever of the two reads it. How chunk scores fold always
 * shows, since every search reads it. How passages grow is its own group
 * (`visibleExpansionFields`).
 */
export function visibleSearchFields(effective: SearchSettings): (keyof SearchSettings)[] {
  const hybrid = effective.mode === 'hybrid'
  const reranked = effective.reranker === 'cross-encoder'
  const fields: (keyof SearchSettings)[] = ['limit', 'mode']
  if (hybrid) fields.push('fusion')
  if (hybrid && effective.fusion === 'rrf') fields.push('rrf_k')
  if (hybrid && effective.fusion === 'linear') fields.push('vector_weight', 'bm25_weight')
  if (effective.mode !== 'fts') fields.push('nprobes', 'refine_factor')
  fields.push('reranker')
  if (reranked) fields.push('reranker_model', 'rerank_with_context', 'min_rerank_score', 'rerank_excerpts')
  if (hybrid || reranked) fields.push('candidates')
  fields.push('score_fold')
  return fields
}

/**
 * Which expansion fields a form shows, in order: how a passage grows, and how large the
 * sections and the answer it grows within may be. Every search reads them, except how a chunk
 * is valued, which only a reranker offers a choice of.
 */
export function visibleExpansionFields(effective: SearchSettings): (keyof SearchSettings)[] {
  const fields: (keyof SearchSettings)[] = ['min_passage_chars', 'max_passage_grow']
  if (effective.reranker === 'cross-encoder') fields.push('fill_values')
  fields.push('grow_bias', 'max_section_chars', 'max_answer_chars')
  return fields
}
