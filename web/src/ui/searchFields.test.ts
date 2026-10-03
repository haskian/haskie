import { describe, expect, test } from 'bun:test'
import type { SearchOverrides, SearchSettings } from '../api'
import { effectiveSearch, visibleExpansionFields, visibleSearchFields } from './searchFields'

// The backend's defaults, as `/api/settings` sends them.
const DEFAULTS: SearchSettings = {
  limit: 10,
  candidates: 50,
  mode: 'hybrid',
  fusion: 'rrf',
  rrf_k: 60,
  vector_weight: 0.7,
  bm25_weight: 0.3,
  nprobes: 20,
  refine_factor: 10,
  reranker: 'none',
  reranker_model: 'cross-encoder/ettin-reranker-32m-v1',
  rerank_with_context: false,
  min_rerank_score: 0.05,
  score_fold: 'sum',
  rerank_excerpts: false,
  fill_values: 'relative',
  min_passage_chars: 300,
  max_passage_grow: 3,
  grow_bias: 0,
  max_section_chars: 12000,
  answer_budget_chars: 36000,
}

// A collection that overrides nothing, as `/api/collections/{name}` sends it.
const NO_OVERRIDES: SearchOverrides = {
  limit: null,
  candidates: null,
  mode: null,
  fusion: null,
  rrf_k: null,
  vector_weight: null,
  bm25_weight: null,
  nprobes: null,
  refine_factor: null,
  reranker: null,
  reranker_model: null,
  rerank_with_context: null,
  min_rerank_score: null,
  score_fold: null,
  rerank_excerpts: null,
  fill_values: null,
  min_passage_chars: null,
  max_passage_grow: null,
  grow_bias: null,
  max_section_chars: null,
  answer_budget_chars: null,
}

const search = (patch: Partial<SearchSettings>): SearchSettings => ({ ...DEFAULTS, ...patch })

describe('effectiveSearch', () => {
  const cases: Array<{ name: string; overrides: SearchOverrides; expected: Partial<SearchSettings> }> = [
    { name: 'no override falls back to the default', overrides: NO_OVERRIDES, expected: DEFAULTS },
    { name: 'an override wins', overrides: { ...NO_OVERRIDES, limit: 25, mode: 'fts' }, expected: { limit: 25, mode: 'fts' } },
    { name: 'zero is a value, not an absent override', overrides: { ...NO_OVERRIDES, bm25_weight: 0 }, expected: { bm25_weight: 0 } },
    { name: 'a negative bias is a value too', overrides: { ...NO_OVERRIDES, grow_bias: -0.5 }, expected: { grow_bias: -0.5 } },
    { name: 'false is a value too', overrides: { ...NO_OVERRIDES, rerank_with_context: false }, expected: { rerank_with_context: false } },
    { name: 'a switch turned on wins', overrides: { ...NO_OVERRIDES, rerank_with_context: true }, expected: { rerank_with_context: true } },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(effectiveSearch(one.overrides, DEFAULTS)).toMatchObject(one.expected)
    })
  }
})

describe('visibleSearchFields', () => {
  const cases: Array<{ name: string; search: SearchSettings; expected: (keyof SearchSettings)[] }> = [
    {
      name: 'hybrid with rrf asks for the constant, the vector knobs and the candidate pool',
      search: DEFAULTS,
      expected: ['limit', 'mode', 'fusion', 'rrf_k', 'nprobes', 'refine_factor', 'reranker', 'candidates', 'score_fold'],
    },
    {
      name: 'linear fusion swaps the constant for the two weights',
      search: search({ fusion: 'linear' }),
      expected: ['limit', 'mode', 'fusion', 'vector_weight', 'bm25_weight', 'nprobes', 'refine_factor', 'reranker', 'candidates', 'score_fold'],
    },
    {
      name: 'a vector query has no fusion and no candidate pool',
      search: search({ mode: 'vector' }),
      expected: ['limit', 'mode', 'nprobes', 'refine_factor', 'reranker', 'score_fold'],
    },
    {
      name: 'full text reads no vector index, so it needs no probes',
      search: search({ mode: 'fts' }),
      expected: ['limit', 'mode', 'reranker', 'score_fold'],
    },
    {
      name: 'a reranker adds its model, and a candidate pool even without hybrid',
      search: search({ mode: 'fts', reranker: 'cross-encoder' }),
      expected: ['limit', 'mode', 'reranker', 'reranker_model', 'rerank_with_context', 'min_rerank_score', 'rerank_excerpts', 'candidates', 'score_fold'],
    },
    {
      name: 'hybrid and a reranker ask for the candidate pool once',
      search: search({ reranker: 'cross-encoder' }),
      expected: ['limit', 'mode', 'fusion', 'rrf_k', 'nprobes', 'refine_factor', 'reranker', 'reranker_model', 'rerank_with_context', 'min_rerank_score', 'rerank_excerpts', 'candidates', 'score_fold'],
    },
    {
      name: 'the order is the order the form renders',
      search: search({ mode: 'hybrid', fusion: 'linear', reranker: 'cross-encoder' }),
      expected: [
        'limit',
        'mode',
        'fusion',
        'vector_weight',
        'bm25_weight',
        'nprobes',
        'refine_factor',
        'reranker',
        'reranker_model',
        'rerank_with_context',
        'min_rerank_score',
        'rerank_excerpts',
        'candidates',
        'score_fold',
      ],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(visibleSearchFields(one.search)).toEqual(one.expected)
    })
  }
})

describe('visibleExpansionFields', () => {
  const cases: Array<{ name: string; search: SearchSettings; expected: (keyof SearchSettings)[] }> = [
    {
      name: 'without a reranker: how passages grow and what they grow within',
      search: DEFAULTS,
      expected: ['min_passage_chars', 'max_passage_grow', 'grow_bias', 'max_section_chars', 'answer_budget_chars'],
    },
    {
      name: 'a reranker adds how a chunk is valued',
      search: search({ reranker: 'cross-encoder' }),
      expected: ['min_passage_chars', 'max_passage_grow', 'fill_values', 'grow_bias', 'max_section_chars', 'answer_budget_chars'],
    },
    {
      name: 'no shortest passage still asks how far passages grow: every excerpt grows by it',
      search: search({ min_passage_chars: 0 }),
      expected: ['min_passage_chars', 'max_passage_grow', 'grow_bias', 'max_section_chars', 'answer_budget_chars'],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(visibleExpansionFields(one.search)).toEqual(one.expected)
    })
  }
})
