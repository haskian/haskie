import { describe, expect, test } from 'bun:test'
import type { EmbedderMetadata, EmbeddingModel } from '../api'
import { count } from '../format'
import { profileLabel, profileOptions } from './options'

// bge-small as `/api/options` sends it: the model under `embedding_profiles`, its metadata apart
const COMPACT: EmbeddingModel = {
  name: 'BAAI/bge-small-en-v1.5',
  dims: 384,
  accelerator: 'auto',
  duplicate: { chunk: 0.92, passage: 0.95 },
  query_prefix: '',
  document_prefix: '',
  matryoshka: null,
}
const METADATA: EmbedderMetadata = {
  description: 'Small and fast; a good default (~130 MB).',
  parameters: 33360512,
  context_tokens: 512,
  languages: 'English',
  license: 'MIT',
  released: '2023-09-12',
  model_card_url: 'https://huggingface.co/BAAI/bge-small-en-v1.5',
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
    { name: 'a model with its metadata: size and parameters', model: COMPACT, metadata: METADATA, expected: 'bge-small-en-v1.5 · Dim 384 · Param 33M' },
    { name: 'a model whose metadata is missing: no parameters', model: COMPACT, metadata: undefined, expected: 'bge-small-en-v1.5 · Dim 384' },
    { name: 'full-text only', model: null, metadata: undefined, expected: 'full-text only' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => expect(profileLabel(testCase.model, testCase.metadata)).toBe(testCase.expected))
  }
})

describe('profileOptions', () => {
  test('each profile is described by its own metadata, and none by what it is', () => {
    const options = profileOptions({ none: null, compact: COMPACT }, { compact: METADATA })

    expect(options).toEqual([
      { value: 'none', label: 'full-text only', sub: 'Full-text search only. No model download.' },
      { value: 'compact', label: 'bge-small-en-v1.5 · Dim 384 · Param 33M', sub: METADATA.description },
    ])
  })
})
