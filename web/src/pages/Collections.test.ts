import { describe, expect, test } from 'bun:test'
import type { CollectionSummary, Document, Member } from '../api'
import { candidateDocuments } from './collections/candidates'
import { groupByName, tileSub } from './collections/group'

// Real payloads: one document, one membership, one collection, as the API answers them.
const DOCUMENT: Document = {
  name: 'area.pdf',
  suffix: '.pdf',
  size: 421_888,
  status: 'imported',
  error: null,
  preview: { kind: 'pdf', truncated: false, pages: 12, ocr_pages: [3, 4] },
  parser: 'anydoc',
  skip_ocr_pages: true,
  created_at: 1_547_901_120,
  updated_at: 1_547_901_180,
  description: 'Notes on area lights and soft shadow falloff.',
  md5: '9e107d9d372bb6826bd81d3542a419d6',
  collections: 1,
}

const document = (name: string): Document => ({ ...DOCUMENT, name })

const member = (name: string): Member => ({
  document: document(name),
  status: 'indexed',
  error: null,
  added_at: 1_547_901_200,
  updated_at: 1_547_901_260,
})

const collection = (name: string, total = 12, active = 0): CollectionSummary => ({
  name,
  counts: { total, indexed: total - active, active, error: 0, by_status: { indexed: total - active, pending: active } },
  created_at: 1_547_900_000,
  description: 'Everything filed under this letter range.',
})

describe('groupByName', () => {
  const cases: Array<{ name: string; value: CollectionSummary[]; expected: Array<[string, string[]]> }> = [
    { name: 'nothing to band', value: [], expected: [] },
    {
      name: 'a band with one collection',
      value: [collection('Area')],
      expected: [['A–E', ['Area']]],
    },
    {
      name: 'empty bands are left out and the rest keep the gallery order',
      value: [collection('Wind'), collection('Area'), collection('Zap')],
      expected: [
        ['A–E', ['Area']],
        ['U–Z', ['Wind', 'Zap']],
      ],
    },
    {
      name: 'names that are not letters band last',
      value: [collection('2019 renders'), collection('Area')],
      expected: [
        ['A–E', ['Area']],
        ['#', ['2019 renders']],
      ],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(groupByName(one.value).map((group) => [group.label, group.items.map((item) => item.name)])).toEqual(one.expected)
    })
  }
})

describe('tileSub', () => {
  const cases: Array<{ name: string; value: CollectionSummary; expected: string }> = [
    { name: 'nothing is being written', value: collection('Area', 12), expected: '12 documents' },
    { name: 'something is being written', value: collection('Area', 12, 3), expected: '12 documents · 3 active' },
    { name: 'an empty collection', value: collection('Area', 0), expected: '0 documents' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(tileSub(one.value)).toBe(one.expected)
    })
  }
})

describe('candidateDocuments', () => {
  const cases: Array<{ name: string; imported: Document[]; members: Member[]; expected: string[] }> = [
    { name: 'nothing imported', imported: [], members: [member('area.pdf')], expected: [] },
    {
      name: 'the collection holds nothing yet',
      imported: [document('area.pdf'), document('box.pdf')],
      members: [],
      expected: ['area.pdf', 'box.pdf'],
    },
    {
      name: 'a held document is not a candidate',
      imported: [document('area.pdf'), document('box.pdf')],
      members: [member('area.pdf')],
      expected: ['box.pdf'],
    },
    {
      name: 'the collection holds everything imported',
      imported: [document('area.pdf')],
      members: [member('area.pdf'), member('gone.pdf')],
      expected: [],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(candidateDocuments(one.imported, one.members).map((doc) => doc.name)).toEqual(one.expected)
    })
  }
})
