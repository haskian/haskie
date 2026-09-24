import { describe, expect, test } from 'bun:test'
import { parseServerTiming, type Hit, type Passage, type SearchScope, type Source, type StepTiming } from '../api'
import { cite, position, seqLabel, type Match } from '../ui/match'
import { parseScope, scopeParams, type Scope } from './explore/scope'

const HIT: Hit = {
  collection: 'A–E',
  document: 'area-lights.pdf',
  source_path: 'documents/area-lights.pdf',
  markdown_path: 'markdown/area-lights.md',
  part: 0,
  seq: 4,
  line_start: 41,
  line_end: 58,
  char_start: 1204,
  char_end: 1702,
  byte_start: 1204,
  byte_end: 1702,
  page_start: 2,
  page_end: 2,
  headings: ['Lighting', 'Soft shadows'],
  frame: ['Lighting', 'Soft shadows'],
  header: 'Lighting > Soft shadows',
  location: 'p. 2',
  text: 'Area lights soften the shadow edge in proportion to their size.',
  layout: [{ type: 'text', position: 0 }],
  start_reason: 'paragraph',
  end_reason: 'length_sentence',
  score: 0.91,
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
  also_in: [],
}

const SOURCE: Source = {
  collection: 'A–E',
  document: 'area-lights.pdf',
  score: 0.91,
  chunks: 7,
  description: 'Notes on area lights and soft shadow falloff.',
  header: 'Lighting > Soft shadows',
  location: 'lines 41–58',
  text: 'Area lights soften the shadow edge in proportion to their size.',
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
  line_start: 41,
  line_end: 58,
  collections: ['A–E'],
  sections: [{ header: 'Lighting > Soft shadows', score: 0.91, chunks: 7, line_start: 41, line_end: 58, location: 'area-lights.pdf p.2 L41-58' }],
}

const PASSAGE: Passage = {
  collection: 'A–E',
  document: 'area-lights.pdf',
  header: 'Lighting > Soft shadows',
  location: 'area-lights.pdf p.2 L41-58',
  seq_start: 4,
  seq_end: 6,
  line_start: 41,
  line_end: 58,
  char_start: 1204,
  char_end: 2402,
  page_start: 2,
  page_end: 2,
  text: 'Area lights soften the shadow edge in proportion to their size.',
  score: 0.91,
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
  also_in: [],
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

describe('scopeParams', () => {
  const cases: Array<{ name: string; value: Scope; expected: SearchScope }> = [
    { name: 'every collection is no filter at all', value: { kind: 'all' }, expected: {} },
    { name: 'one collection filters on itself', value: { kind: 'collection', name: 'A–E' }, expected: { collections: ['A–E'] } },
    { name: 'a session is sent by id, and the backend reads its selection', value: { kind: 'session', id: 'agent-1' }, expected: { session_id: 'agent-1' } },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(scopeParams(testCase.value)).toEqual(testCase.expected)
    })
  }
})

describe('position', () => {
  const cases: Array<{ name: string; value: Match; expected: string }> = [
    { name: 'a paged chunk reads as its page and chunk', value: HIT, expected: 'p. 2 · chunk 4' },
    { name: 'page zero is a page, not a missing one', value: { ...HIT, page_start: 0, page_end: 0 }, expected: 'p. 0 · chunk 4' },
    { name: 'a chunk across two pages names both', value: { ...HIT, page_end: 3 }, expected: 'p. 2–3 · chunk 4' },
    { name: 'a chunk with no pages is its chunk alone', value: { ...HIT, page_start: null, page_end: null }, expected: 'chunk 4' },
    { name: 'a passage reads as its page, its lines and its chunks', value: PASSAGE, expected: 'p. 2 · lines 41–58 · chunks 4–6' },
    { name: 'a passage of one line and one chunk, without pages', value: { ...PASSAGE, line_end: 41, seq_end: 4, page_start: null, page_end: null }, expected: 'line 41 · chunk 4' },
    { name: 'a source has only lines', value: SOURCE, expected: 'lines 41–58' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(position(testCase.value)).toBe(testCase.expected)
    })
  }
})

describe('seqLabel', () => {
  const cases: Array<{ name: string; value: Match; expected: string }> = [
    { name: 'a hit is its own sequence number', value: HIT, expected: '4' },
    { name: 'a passage over several chunks is the run', value: PASSAGE, expected: '4–6' },
    { name: 'a passage of one chunk is that number alone', value: { ...PASSAGE, seq_end: 4 }, expected: '4' },
    { name: 'a source shows no quote, so no number', value: SOURCE, expected: '' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(seqLabel(testCase.value)).toBe(testCase.expected)
    })
  }
})

describe('cite', () => {
  const cases: Array<{ name: string; location: string; doc: string; expected: string }> = [
    { name: 'a PDF drops the document name, keeping page and lines', location: 'area-lights.pdf p.2 L41-58', doc: 'area-lights.pdf', expected: 'p.2 L41-58' },
    { name: 'a non-PDF keeps only its lines', location: 'notes.md L7-43', doc: 'notes.md', expected: 'L7-43' },
    { name: 'a location that does not start with the name is left whole', location: 'p.2 L41-58', doc: 'area-lights.pdf', expected: 'p.2 L41-58' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(cite(testCase.location, testCase.doc)).toBe(testCase.expected)
    })
  }
})

describe('parseServerTiming', () => {
  const cases: Array<{ name: string; header: string | null; expected: StepTiming[] }> = [
    { name: 'no header, no steps', header: null, expected: [] },
    {
      name: 'every step with its time and its name, in order',
      header: 'plan;dur=12.3;desc="Embed the query", retrieve;dur=41.2;desc="LanceDB retrieval"',
      expected: [
        { step: 'plan', label: 'Embed the query', ms: 12.3 },
        { step: 'retrieve', label: 'LanceDB retrieval', ms: 41.2 },
      ],
    },
    { name: 'a step with no desc is named by itself', header: 'merge;dur=0.4', expected: [{ step: 'merge', label: 'merge', ms: 0.4 }] },
  ]
  for (const { name, header, expected } of cases) test(name, () => expect(parseServerTiming(header)).toEqual(expected))
})
