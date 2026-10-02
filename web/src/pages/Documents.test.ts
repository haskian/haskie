import { describe, expect, test } from 'bun:test'
import type { Document, DocumentStatus, EmbeddingEntry, ImportedDocument, Staged } from '../api'
import { embeddingLabel } from './documents/embedding'
import { groupByDay, groupByStatus, unfiledFirst } from './documents/group'
import { fresh, importedNames, importLabel, staged, waitingAfter, type StagedFile } from './documents/staged'

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
  id: '9e107d9d372bb6826bd81d3542a419d6',
  collections: ['lighting', 'rendering'],
}

const doc = (over: Partial<Document>): Document => ({ ...DOC, ...over })

// `document_statuses` as `/api/options` sends it: the import pipeline's order.
const STATUSES: DocumentStatus[] = ['queued', 'converting', 'embedding', 'describing', 'imported', 'error', 'cancelled', 'deleting']

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
      name: 'in a day, a document in no collection comes before a newer one in a collection',
      value: [doc({ name: 'Area' }), doc({ name: 'Box', created_at: DOC.created_at - 60, collections: [] })],
      expected: [['Sat 19 Jan', ['Box', 'Area']]],
    },
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

describe('unfiledFirst', () => {
  const cases: Array<{ name: string; value: Document[]; expected: string[] }> = [
    { name: 'no documents', value: [], expected: [] },
    {
      name: 'in no collection first, whatever the date',
      value: [doc({ name: 'Filed', created_at: DOC.created_at + 60 }), doc({ name: 'Unfiled', collections: [] })],
      expected: ['Unfiled', 'Filed'],
    },
    {
      name: 'newest first among the filed and among the unfiled',
      value: [
        doc({ name: 'Old filed' }),
        doc({ name: 'Old unfiled', collections: [] }),
        doc({ name: 'New filed', created_at: DOC.created_at + 60 }),
        doc({ name: 'New unfiled', created_at: DOC.created_at + 60, collections: [] }),
      ],
      expected: ['New unfiled', 'Old unfiled', 'New filed', 'Old filed'],
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect([...testCase.value].sort(unfiledFirst).map((one) => one.name)).toEqual(testCase.expected)
    })
  }
})

describe('embeddingLabel', () => {
  // A real row of `/api/documents/{document}/embeddings`.
  const ENTRY: EmbeddingEntry = {
    document_id: '9e107d9d372bb6826bd81d3542a419d6',
    model: 'BAAI/bge-small-en-v1.5',
    chunk_size: 1200,
    chunk_merge_below: 33,
    chunk_frame: true,
    chunker: 'markdown',
    chunk_version: 1,
    parser: 'anydoc',
    skip_ocr_pages: true,
    id: 'a62894d5043251b553a18ba785281ca5e3ba8a28e58317a2f4207838f219e370',
    urn: 'document_id:9e107d9d372bb6826bd81d3542a419d6;model:BAAI/bge-small-en-v1.5;chunk_size:1200;chunk_merge_below:33;chunk_frame:true;chunker:markdown;chunk_version:1;parser:anydoc;skip_ocr_pages:true',
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

describe('staging several files', () => {
  // `POST /api/documents/staging` as it answers for a real upload.
  const STAGED: Staged = { staging_id: '3f2b9c1e8a7d4b6f', filename: 'Area Lights.PDF', name: 'area-lights.pdf', size: 421_904, duplicate: null }
  const file = (name: string) => new File(['%PDF-1.4'], name, { type: 'application/pdf' })
  const cases: Array<{
    name: string
    files: File[]
    results: PromiseSettledResult<Staged>[]
    expected: { added: Array<[string, string | null]>; failures: string[] }
  }> = [
    { name: 'nothing picked stages nothing', files: [], results: [], expected: { added: [], failures: [] } },
    {
      name: 'each landed file takes the name the server normalized, a repeat keeps the document it already is',
      files: [file('Area Lights.PDF'), file('Soft Shadows.pdf')],
      results: [
        { status: 'fulfilled', value: STAGED },
        { status: 'fulfilled', value: { ...STAGED, staging_id: '9a1c', filename: 'Soft Shadows.pdf', name: 'soft-shadows.pdf', duplicate: 'shadows.pdf' } },
      ],
      expected: { added: [['area-lights.pdf', null], ['soft-shadows.pdf', 'shadows.pdf']], failures: [] },
    },
    {
      name: 'a refused file costs only itself, and says which one it was',
      files: [file('Area Lights.PDF'), file('huge.pdf')],
      results: [
        { status: 'fulfilled', value: STAGED },
        { status: 'rejected', reason: new Error('Request Entity Too Large') },
      ],
      expected: { added: [['area-lights.pdf', null]], failures: ['huge.pdf: Request Entity Too Large'] },
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      const got = staged(testCase.files, testCase.results)
      expect({ added: got.added.map((one) => [one.name, one.duplicate]), failures: got.failures }).toEqual(testCase.expected)
      expect(got.added.every((one) => one.error === null)).toBe(true)
    })
  }
})

describe('importing the staged set', () => {
  const waiting: StagedFile[] = [
    { staging_id: 'a1', filename: 'area-lights.pdf', size: 421_904, duplicate: null, name: 'Area lights', error: null },
    { staging_id: 'b2', filename: 'notes.md', size: 2_048, duplicate: null, name: 'notes.md', error: 'an older refusal' },
  ]
  // `POST /api/documents/import` answers with the document row, without the listing's collections.
  const { collections: _held, ...IMPORTED } = DOC
  const row = (name: string): ImportedDocument => ({ ...IMPORTED, name, status: 'queued' })
  const cases: Array<{
    name: string
    results: PromiseSettledResult<ImportedDocument>[]
    expected: { waiting: Array<[string, string | null]>; imported: string[] }
  }> = [
    {
      name: 'every file imported empties the set',
      results: [
        { status: 'fulfilled', value: row('Area lights') },
        { status: 'fulfilled', value: row('notes.md') },
      ],
      expected: { waiting: [], imported: ['Area lights', 'notes.md'] },
    },
    {
      name: 'a refused file stays, with the reason, while the rest import',
      results: [
        { status: 'fulfilled', value: row('Area lights') },
        { status: 'rejected', reason: new Error("a document named 'notes.md' already exists") },
      ],
      expected: { waiting: [['notes.md', "a document named 'notes.md' already exists"]], imported: ['Area lights'] },
    },
    {
      name: 'every file refused keeps the whole set',
      results: [
        { status: 'rejected', reason: new Error('staging expired') },
        { status: 'rejected', reason: 'offline' },
      ],
      expected: { waiting: [['Area lights', 'staging expired'], ['notes.md', 'offline']], imported: [] },
    },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      const left = waitingAfter(waiting, waiting, testCase.results)
      expect({ waiting: left.map((one) => [one.name, one.error]), imported: importedNames(testCase.results) }).toEqual(testCase.expected)
    })
  }
  test('a file staged and a name edited while the imports ran are kept as they are now', () => {
    const late: StagedFile = { staging_id: 'c3', filename: 'grinding.md', size: 512, duplicate: null, name: 'grinding.md', error: null }
    const now = [{ ...waiting[1], name: 'Notes on brewing' }, late] // the first file left, the second renamed
    const left = waitingAfter(now, waiting, [
      { status: 'fulfilled', value: row('Area lights') },
      { status: 'rejected', reason: new Error("a document named 'notes.md' already exists") },
    ])
    expect(left.map((one) => [one.name, one.error])).toEqual([
      ['Notes on brewing', "a document named 'notes.md' already exists"],
      ['grinding.md', null],
    ])
  })
})

describe('fresh', () => {
  const file = (staging_id: string, duplicate: string | null): StagedFile => ({
    staging_id,
    filename: `${staging_id}.md`,
    size: 2_048,
    duplicate,
    name: `${staging_id}.md`,
    error: null,
  })
  const cases: Array<[string, StagedFile[], string[]]> = [
    ['nothing staged sends nothing', [], []],
    ['new bytes are all sent', [file('a1', null), file('b2', null)], ['a1', 'b2']],
    ['a file already imported is left out', [file('a1', null), file('b2', 'guide.md')], ['a1']],
    ['every file already imported sends nothing', [file('a1', 'guide.md')], []],
  ]
  for (const [name, files, expected] of cases) {
    test(name, () => expect(fresh(files).map((one) => one.staging_id)).toEqual(expected))
  }
})

describe('importLabel', () => {
  const cases: Array<[string, number, string]> = [
    ['one file is a plain Import', 1, 'Import'],
    ['several say how many', 3, 'Import 3 documents'],
  ]
  for (const [name, count, expected] of cases) test(name, () => expect(importLabel(count)).toBe(expected))
})
