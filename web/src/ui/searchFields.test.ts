import { describe, expect, test } from 'bun:test'
import type { SearchOverrides, SearchSettings } from '../api'
import { effectiveSearch, visibleSearchFields } from './searchFields'

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
  reranker_model: 'Xenova/ms-marco-MiniLM-L-6-v2',
  rerank_with_context: false,
  min_rerank_score: 0.05,
  min_passage_chars: 300,
  max_passage_grow: 2,
  max_section_chars: 8000,
  max_answer_chars: 24000,
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
  min_passage_chars: null,
  max_passage_grow: null,
  max_section_chars: null,
  max_answer_chars: null,
}

const search = (patch: Partial<SearchSettings>): SearchSettings => ({ ...DEFAULTS, ...patch })

describe('effectiveSearch', () => {
  const cases: Array<{ name: string; overrides: SearchOverrides; expected: Partial<SearchSettings> }> = [
    { name: 'no override falls back to the default', overrides: NO_OVERRIDES, expected: DEFAULTS },
    { name: 'an override wins', overrides: { ...NO_OVERRIDES, limit: 25, mode: 'fts' }, expected: { limit: 25, mode: 'fts' } },
    { name: 'zero is a value, not an absent override', overrides: { ...NO_OVERRIDES, bm25_weight: 0 }, expected: { bm25_weight: 0 } },
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
      expected: ['limit', 'mode', 'fusion', 'rrf_k', 'nprobes', 'refine_factor', 'reranker', 'candidates', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'linear fusion swaps the constant for the two weights',
      search: search({ fusion: 'linear' }),
      expected: ['limit', 'mode', 'fusion', 'vector_weight', 'bm25_weight', 'nprobes', 'refine_factor', 'reranker', 'candidates', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'a vector query has no fusion and no candidate pool',
      search: search({ mode: 'vector' }),
      expected: ['limit', 'mode', 'nprobes', 'refine_factor', 'reranker', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'full text reads no vector index, so it needs no probes',
      search: search({ mode: 'fts' }),
      expected: ['limit', 'mode', 'reranker', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'a reranker adds its model, and a candidate pool even without hybrid',
      search: search({ mode: 'fts', reranker: 'cross-encoder' }),
      expected: ['limit', 'mode', 'reranker', 'reranker_model', 'rerank_with_context', 'min_rerank_score', 'candidates', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'hybrid and a reranker ask for the candidate pool once',
      search: search({ reranker: 'cross-encoder' }),
      expected: ['limit', 'mode', 'fusion', 'rrf_k', 'nprobes', 'refine_factor', 'reranker', 'reranker_model', 'rerank_with_context', 'min_rerank_score', 'candidates', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
    },
    {
      name: 'no shortest passage still asks how far passages grow: every excerpt grows by it',
      search: search({ mode: 'fts', min_passage_chars: 0 }),
      expected: ['limit', 'mode', 'reranker', 'min_passage_chars', 'max_passage_grow', 'max_section_chars', 'max_answer_chars'],
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
        'candidates',
        'min_passage_chars',
        'max_passage_grow',
        'max_section_chars',
        'max_answer_chars',
      ],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(visibleSearchFields(one.search)).toEqual(one.expected)
    })
  }
})
