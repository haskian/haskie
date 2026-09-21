import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import type { DocumentMatch, Hit } from '../api'
import { position, type Match } from '../ui/match'
import { parseScope, scopeCollections, searchSections, type Scope } from './explore/scope'

// The sessions the picker knows about, as `/api/sessions` answers them.
const SESSIONS: Record<string, string[]> = { 'agent-1': ['A–E', 'K–O'], empty: [] }

const HIT: Hit = {
  collection: 'A–E',
  doc: 'area-lights.pdf',
  source_path: 'documents/area-lights.pdf',
  markdown_path: 'markdown/area-lights.md',
  part: 0,
  chunk_id: 3,
  line_start: 41,
  line_end: 58,
  char_start: 1204,
  char_end: 1702,
  page_start: 2,
  page_end: 2,
  parents: ['Lighting'],
  heading: 'Lighting › Soft shadows',
  header: 'Soft shadows',
  location: 'p. 2',
  text: 'Area lights soften the shadow edge in proportion to their size.',
  score: 0.91,
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
}

const MATCH: DocumentMatch = {
  collection: 'A–E',
  doc: 'area-lights.pdf',
  score: 0.91,
  chunks: 7,
  description: 'Notes on area lights and soft shadow falloff.',
  heading: 'Lighting › Soft shadows',
  location: 'lines 41–58',
  text: 'Area lights soften the shadow edge in proportion to their size.',
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
  line_start: 41,
  line_end: 58,
}

describe('parseScope', () => {
  const cases: Array<{ name: string; value: string; expected: Scope }> = [
    { name: 'the star is every collection', value: '*', expected: { kind: 'all' } },
    { name: 'a plain name is a collection', value: 'A–E', expected: { kind: 'collection', name: 'A–E' } },
    { name: 'the prefix names a session', value: 'session:agent-1', expected: { kind: 'session', id: 'agent-1' } },
    { name: 'a session id may be empty after the prefix', value: 'session:', expected: { kind: 'session', id: '' } },
    { name: 'a collection whose name holds a colon is still a collection', value: 'notes:2019', expected: { kind: 'collection', name: 'notes:2019' } },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(parseScope(testCase.value)).toEqual(testCase.expected)
    })
  }
})

describe('scopeCollections', () => {
  const cases: Array<{ name: string; value: Scope; expected: string[] | undefined }> = [
    { name: 'every collection is no filter at all', value: { kind: 'all' }, expected: undefined },
    { name: 'one collection filters on itself', value: { kind: 'collection', name: 'A–E' }, expected: ['A–E'] },
    { name: 'a session filters on the collections it set', value: { kind: 'session', id: 'agent-1' }, expected: ['A–E', 'K–O'] },
    { name: 'a session that set none matches nothing', value: { kind: 'session', id: 'empty' }, expected: [] },
    { name: 'an unknown session matches nothing', value: { kind: 'session', id: 'gone' }, expected: [] },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(scopeCollections(testCase.value, SESSIONS)).toEqual(testCase.expected)
    })
  }
})

describe('position', () => {
  const cases: Array<{ name: string; value: Match; expected: string }> = [
    { name: 'a paged hit reads as page and chunk', value: HIT, expected: 'p. 2 · chunk 3' },
    { name: 'page zero is a page, not a missing one', value: { ...HIT, page_start: 0 }, expected: 'p. 0 · chunk 3' },
    { name: 'a hit with no pages falls back to lines', value: { ...HIT, page_start: null }, expected: 'lines 41–58' },
    { name: 'a document match has only lines', value: MATCH, expected: 'lines 41–58' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(position(testCase.value)).toBe(testCase.expected)
    })
  }
})

// `searchSections` picks one of three endpoints. The request itself is the behaviour under test,
// so `fetch` is replaced with a recorder that answers each endpoint's own shape.
describe('searchSections', () => {
  const realFetch = globalThis.fetch
  let requested: string[] = []

  beforeEach(() => {
    requested = []
    globalThis.fetch = ((input: RequestInfo | URL) => {
      const url = String(input)
      requested.push(url)
      // /api/search/text is paged; the other two answer a bare array.
      const body = url.startsWith('/api/search/text') ? { items: [HIT], next_cursor: null, total: 1 } : [HIT]
      return Promise.resolve(new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } }))
    }) as typeof fetch
  })
  afterEach(() => {
    globalThis.fetch = realFetch
  })

  const cases: Array<{ name: string; value: Scope; expected: string }> = [
    { name: 'every collection goes to the full-text endpoint', value: { kind: 'all' }, expected: '/api/search/text?page_size=50&q=shadow' },
    { name: 'one collection searches that collection', value: { kind: 'collection', name: 'A–E' }, expected: '/api/collections/A%E2%80%93E/search?q=shadow' },
    { name: 'a session searches through the session', value: { kind: 'session', id: 'agent-1' }, expected: '/api/search?session_id=agent-1&q=shadow' },
  ]
  for (const testCase of cases) {
    test(testCase.name, async () => {
      const hits = await searchSections(testCase.value, 'shadow')
      expect(requested).toEqual([testCase.expected])
      expect(hits).toEqual([HIT]) // the paged answer is unwrapped to the same list as the others
    })
  }
})
