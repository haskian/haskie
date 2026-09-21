import { describe, expect, test } from 'bun:test'
import type { Document } from '../api'
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
  collections: 2,
}

const doc = (over: Partial<Document>): Document => ({ ...DOC, ...over })

// Both groupings answer with labels and names, which is what the gallery renders.
const shape = (groups: Array<{ label: string; items: Document[] }>): Array<[string, string[]]> =>
  groups.map((group) => [group.label, group.items.map((item) => item.name)])

describe('groupByStatus', () => {
  const cases: Array<{ name: string; value: Document[]; expected: Array<[string, string[]]> }> = [
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
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(shape(groupByStatus(testCase.value))).toEqual(testCase.expected)
    })
  }
})

describe('groupByDay', () => {
  const DAY = 86_400
  const cases: Array<{ name: string; value: Document[]; expected: Array<[string, string[]]> }> = [
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
