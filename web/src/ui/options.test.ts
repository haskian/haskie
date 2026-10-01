import { describe, expect, test } from 'bun:test'
import type { EmbedderMetadata, EmbeddingModel } from '../api'
import { count } from '../format'
import { profileLabel, profileOptions } from './options'

// granite-97m as `/api/options` sends it: the model under `embedding_profiles`, its metadata apart
const GRANITE: EmbeddingModel = {
  name: 'ibm-granite/granite-embedding-97m-multilingual-r2',
  dims: 384,
  accelerator: 'auto',
  duplicate: null,
  profile: 'granite-97m-multilingual',
  weak_match: 0.79,
  answered_match: 0.885,
  same_topic: 0.82,
  query_prefix: '',
  document_prefix: '',
  matryoshka: false,
}
const METADATA: EmbedderMetadata = {
  description: 'The best all-round small multilingual embedder: #1 on multilingual and reasoning retrieval; a good default (~390 MB).',
  parameters: 97441152,
  context_tokens: 32768,
  languages: 'multilingual (200+, 52 enhanced)',
  license: 'Apache-2.0',
  released: '2026-04-20',
  model_card_url: 'https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2',
  runtime: 'onnx',
  devices: ['cpu', 'apple_silicon', 'gpu'],
  dimensions: 384,
}

describe('count', () => {
  const cases: Array<{ name: string; value: number; expected: string }> = [
    { name: 'below a thousand, as it is', value: 512, expected: '512' },
    { name: 'thousands, one decimal where it tells', value: 8192, expected: '8.2K' },
    { name: 'thousands, whole', value: 131072, expected: '131K' },
    { name: 'millions, rounded', value: 33360512, expected: '33M' },
    { name: 'billions, one decimal', value: 1_209_000_000, expected: '1.2B' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => expect(count(testCase.value)).toBe(testCase.expected))
  }
})

describe('profileLabel', () => {
  const cases: Array<{ name: string; model: EmbeddingModel | null; metadata: EmbedderMetadata | undefined; expected: string }> = [
    { name: 'a model with its metadata: size and parameters', model: GRANITE, metadata: METADATA, expected: 'granite-embedding-97m-multilingual-r2 · Dim 384 · Param 97M' },
    { name: 'a model whose metadata is missing: no parameters', model: GRANITE, metadata: undefined, expected: 'granite-embedding-97m-multilingual-r2 · Dim 384' },
    { name: 'full-text only', model: null, metadata: undefined, expected: 'full-text only' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => expect(profileLabel(testCase.model, testCase.metadata)).toBe(testCase.expected))
  }
})

describe('profileOptions', () => {
  test('each profile is described by its own metadata, and none by what it is', () => {
    const options = profileOptions({ none: null, 'granite-97m-multilingual': GRANITE }, { 'granite-97m-multilingual': METADATA })

    expect(options).toEqual([
      { value: 'none', label: 'full-text only', sub: 'Full-text search only. No model download.' },
      { value: 'granite-97m-multilingual', label: 'granite-embedding-97m-multilingual-r2 · Dim 384 · Param 97M', sub: METADATA.description },
    ])
  })
})
