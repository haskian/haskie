import { describe, expect, test } from 'bun:test'
import type { Document, DocumentStatus, EmbeddingEntry } from '../api'
import { embeddingLabel } from './documents/embedding'
import { groupByDay, groupByStatus } from './documents/group'

// One real row, overridden per case: the listing hands the page whole documents, so the fixtures do too.
const DOC: Document = {
  name: 'area-lights.pdf',
  suffix: '.pdf',
  size: 421_904,
  status: 'imported',
  error: null,
  preview: { kind: 'pdf', truncated: false, pages: 3, ocr_pages: [] },
  parser: 'anydoc',
  skip_ocr_pages: true,
  created_at: 1_547_907_120,
  updated_at: 1_547_907_180,
  description: 'Notes on area lights and soft shadow falloff.',
  md5: '9e107d9d372bb6826bd81d3542a419d6',
  collections: 2,
}

const doc = (over: Partial<Document>): Document => ({ ...DOC, ...over })

// `document_statuses` as `/api/options` sends it: the import pipeline's order.
const STATUSES: DocumentStatus[] = ['queued', 'converting', 'embedding', 'imported', 'error', 'cancelled', 'deleting']

// Both groupings answer with labels and names, which is what the gallery renders.
const shape = (groups: Array<{ label: string; items: Document[] }>): Array<[string, string[]]> =>
  groups.map((group) => [group.label, group.items.map((item) => item.name)])

describe('groupByStatus', () => {
  const cases: Array<{ name: string; value: Document[]; statuses?: DocumentStatus[]; expected: Array<[string, string[]]> }> = [
    { name: 'no documents, no bands', value: [], expected: [] },
    {
      name: 'one band per status present, capitalised',
      value: [doc({ name: 'Area' }), doc({ name: 'Lamp', status: 'queued' })],
      expected: [
        ['Queued', ['Lamp']],
        ['Imported', ['Area']],
      ],
    },
    {
      name: 'bands follow the pipeline order, not the rows',
      value: [doc({ name: 'Zap', status: 'error' }), doc({ name: 'Area' }), doc({ name: 'Box', status: 'converting' })],
      expected: [
        ['Converting', ['Box']],
        ['Imported', ['Area']],
        ['Error', ['Zap']],
      ],
    },
    {
      name: 'a status with several documents keeps the listing order',
      value: [doc({ name: 'Cone' }), doc({ name: 'Area' })],
      expected: [['Imported', ['Cone', 'Area']]],
    },
    {
      name: 'the backend decides the order, and a status it does not list has no band',
      value: [doc({ name: 'Area' }), doc({ name: 'Box', status: 'error' }), doc({ name: 'Lamp', status: 'queued' })],
      statuses: ['error', 'imported'],
      expected: [
        ['Error', ['Box']],
        ['Imported', ['Area']],
      ],
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(shape(groupByStatus(testCase.value, testCase.statuses ?? STATUSES))).toEqual(testCase.expected)
    })
  }
})

describe('groupByDay', () => {
  const DAY = 86_400
  const cases: Array<{ name: string; value: Document[]; statuses?: DocumentStatus[]; expected: Array<[string, string[]]> }> = [
    { name: 'no documents, no bands', value: [], expected: [] },
    { name: 'one day, one band', value: [doc({ name: 'Area' })], expected: [['Sat 19 Jan', ['Area']]] },
    {
      name: 'newest day first, newest import first within it',
      value: [doc({ name: 'Area' }), doc({ name: 'Box', created_at: DOC.created_at - DAY }), doc({ name: 'Cone', created_at: DOC.created_at + 60 })],
      expected: [
        ['Sat 19 Jan', ['Cone', 'Area']],
        ['Fri 18 Jan', ['Box']],
      ],
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(shape(groupByDay(testCase.value))).toEqual(testCase.expected)
    })
  }
})

describe('embeddingLabel', () => {
  // A real row of `/api/documents/{document}/embeddings`.
  const ENTRY: EmbeddingEntry = {
    document: 'area-lights.pdf',
    model: 'BAAI/bge-small-en-v1.5',
    chunk_size: 1200,
    chunk_merge_below: 33,
    chunk_frame: true,
    chunker: 'markdown',
    chunk_version: 1,
    parser: 'anydoc',
    skip_ocr_pages: true,
    id: 'a62894d5043251b553a18ba785281ca5e3ba8a28e58317a2f4207838f219e370',
    urn: 'document:area-lights.pdf;model:BAAI/bge-small-en-v1.5;chunk_size:1200;chunk_merge_below:33;chunk_frame:true;chunker:markdown;chunk_version:1;parser:anydoc;skip_ocr_pages:true',
    rows: 2009,
    bytes: 3_402_112,
    created_at: 1_547_907_180,
  }
  const cases: Array<{ name: string; value: EmbeddingEntry; expected: string }> = [
    { name: 'every chunk setting of the cache id, then the rows', value: ENTRY, expected: 'BAAI/bge-small-en-v1.5 · markdown 1200 · merge 33% · frame · 2009 rows' },
    {
      name: 'an entry without the frame says so, so it differs from one with it',
      value: { ...ENTRY, chunk_frame: false },
      expected: 'BAAI/bge-small-en-v1.5 · markdown 1200 · merge 33% · no frame · 2009 rows',
    },
    {
      name: 'a text chunker that never merges and a profile with no model',
      value: { ...ENTRY, model: 'none', chunker: 'text', chunk_size: 400, chunk_merge_below: 0, rows: 1 },
      expected: 'none · text 400 · merge 0% · frame · 1 rows',
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(embeddingLabel(testCase.value)).toBe(testCase.expected)
    })
  }
})
