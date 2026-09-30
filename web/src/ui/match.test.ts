import { describe, expect, test } from 'bun:test'
import type { Excerpt, Hit, Passage } from '../api'
import { alsoDocuments, alsoOf, questionLabels, chunkSizes, everyPlace, fillOf, frameOf, headingOf, isExcerpt, MIN_FILL, overlapHint, pieceMeta, piecesOf, seqLabel, type Reference } from './match'

describe('fillOf', () => {
  const cases: Array<{ name: string; score: number; scores: number[]; expected: number }> = [
    { name: 'the best result fills the bar', score: 0.9, scores: [0.9, 0.5, 0.2], expected: 1 },
    { name: 'the weakest keeps the minimum', score: 0.2, scores: [0.9, 0.5, 0.2], expected: MIN_FILL },
    { name: 'the rest sit between, by score', score: 0.55, scores: [0.9, 0.55, 0.2], expected: 0.55 },
    { name: 'one result fills', score: 0.3, scores: [0.3], expected: 1 },
    { name: 'a tie fills every bar', score: 0.3, scores: [0.3, 0.3], expected: 1 },
    { name: 'raw BM25 scores above one still rank', score: 4, scores: [12, 4], expected: MIN_FILL },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(fillOf(one.score, Math.max(...one.scores), Math.min(...one.scores))).toBeCloseTo(one.expected, 10)
    })
  }
})

// Three whole sentences, as the chunker packs them. The layout marks where each one starts.
const CHUNK: Hit = {
  collection: 'A–E',
  document_id: 'a1',
  document: 'area-lights.pdf',
  source_path: 'documents/area-lights.pdf',
  markdown_path: 'markdown/area-lights.md',
  part: 0,
  seq: 4,
  line_start: 41,
  line_end: 43,
  char_start: 1204,
  char_end: 1330,
  byte_start: 1204,
  byte_end: 1330,
  page_start: 2,
  page_end: 2,
  headings: ['Lighting', 'Soft shadows'],
  frame: ['Lighting', 'Soft shadows'],
  header: 'Lighting > Soft shadows',
  location: 'p. 2',
  text: 'A point light casts a hard edge. Area lights soften it in proportion to their size. Sky light softens it most.',
  layout: [
    { type: 'text', position: 0 },
    { type: 'text', position: 33 },
    { type: 'text', position: 84 },
  ],
  start_reason: 'paragraph',
  end_reason: 'length_sentence',
  score: 0.91,
  source_file: '/Users/ada/.haskie/documents/area-lights.pdf',
  markdown_file: '/Users/ada/.haskie/markdown/area-lights.md',
  also_in: [],
}

const [FIRST, SECOND, THIRD] = ['A point light casts a hard edge. ', 'Area lights soften it in proportion to their size. ', 'Sky light softens it most.']

describe('headingOf', () => {
  const cases: Array<{ name: string; hit: Hit; expected: string }> = [
    { name: 'a chunk sits under the last of its headings', hit: CHUNK, expected: 'Soft shadows' },
    { name: 'a chunk before the first heading sits under none', hit: { ...CHUNK, headings: [] }, expected: '' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(headingOf(one.hit)).toBe(one.expected)
    })
  }
})

describe('piecesOf', () => {
  const cases: Array<{ name: string; hit: Hit; expected: Array<[string, string, number]> }> = [
    {
      name: 'the layout cuts the text back into its typed pieces',
      hit: CHUNK,
      expected: [
        ['text', FIRST, 0],
        ['text', SECOND, 33],
        ['text', THIRD, 84],
      ],
    },
    {
      name: 'a list item, then a code block',
      hit: {
        ...CHUNK,
        text: '- Soften it.\n\n```\nlight()\n```',
        layout: [
          { type: 'list', position: 0 },
          { type: 'code', position: 14 },
        ],
      },
      expected: [
        ['list', '- Soften it.\n\n', 0],
        ['code', '```\nlight()\n```', 14],
      ],
    },
    { name: 'no layout: the text is one piece of text', hit: { ...CHUNK, layout: [] }, expected: [['text', CHUNK.text, 0]] },
    {
      name: 'text ahead of the first position is plain text',
      hit: { ...CHUNK, text: 'Lead. | a |', layout: [{ type: 'table', position: 6 }] },
      expected: [
        ['text', 'Lead. ', 0],
        ['table', '| a |', 6],
      ],
    },
    {
      // counted in code points, as Python counts them: one emoji is one character, two UTF-16 units
      name: 'characters outside the BMP count once',
      hit: {
        ...CHUNK,
        text: '🔦 on. Soft. 🌤 off.',
        layout: [
          { type: 'list', position: 0 },
          { type: 'list', position: 6 },
          { type: 'quote', position: 12 },
        ],
      },
      expected: [
        ['list', '🔦 on. ', 0],
        ['list', 'Soft. ', 6],
        ['quote', '🌤 off.', 12],
      ],
    },
    {
      name: 'a position past the text is ignored, a stale row cannot cut outside it',
      hit: {
        ...CHUNK,
        text: 'One. Two.',
        layout: [
          { type: 'text', position: 0 },
          { type: 'text', position: 5 },
          { type: 'code', position: 40 },
        ],
      },
      expected: [
        ['text', 'One. ', 0],
        ['text', 'Two.', 5],
      ],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(piecesOf(one.hit).map((piece) => [piece.type, piece.text, piece.position])).toEqual(one.expected)
    })
  }
  test('the hint under a piece says where it starts and how big it is', () => {
    expect(pieceMeta({ type: 'text', text: 'Area lights soften it. ', position: 17 })).toBe('position 17 · 23 chars · 4 words')
  })
})

describe('chunkSizes', () => {
  const cases: Array<{ name: string; hit: Hit; expected: ReturnType<typeof chunkSizes> }> = [
    {
      name: 'the heading path and the text, and the two together as embedded',
      hit: { ...CHUNK, frame: ['Lighting', 'Soft shadows'] },
      expected: {
        // "Lighting > Soft shadows\n\n": 25 characters, 3 words, one line
        frame: { chars: 25, words: 3, pieces: 1 },
        text: { chars: Array.from(CHUNK.text).length, words: 21, pieces: 3 },
        total: { chars: 25 + Array.from(CHUNK.text).length, words: 24, pieces: 4 },
      },
    },
    {
      name: 'text before the first heading has no frame',
      hit: { ...CHUNK, frame: [] },
      expected: {
        frame: { chars: 0, words: 0, pieces: 0 },
        text: { chars: Array.from(CHUNK.text).length, words: 21, pieces: 3 },
        total: { chars: Array.from(CHUNK.text).length, words: 21, pieces: 3 },
      },
    },
    {
      name: 'Chinese words are counted without spaces',
      hit: { ...CHUNK, frame: [], text: '我們需要一個數據庫。', layout: [{ type: 'text', position: 0 }] },
      expected: {
        frame: { chars: 0, words: 0, pieces: 0 },
        text: { chars: 10, words: expect.any(Number) as unknown as number, pieces: 1 },
        total: { chars: 10, words: expect.any(Number) as unknown as number, pieces: 1 },
      },
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(chunkSizes(one.hit)).toEqual(one.expected)
    })
  }
  test('a Chinese sentence is more than one word', () => {
    expect(chunkSizes({ ...CHUNK, frame: [], text: '我們需要一個數據庫。', layout: [{ type: 'text', position: 0 }] }).text.words).toBeGreaterThan(1)
  })
  test('the frame is the heading path it was embedded under and a blank line, or nothing', () => {
    expect(frameOf({ ...CHUNK, frame: ['A', 'B'] })).toBe('A > B\n\n')
    expect(frameOf({ ...CHUNK, headings: ['Book', 'A', 'B'], frame: ['A', 'B'] })).toBe('A > B\n\n')
    expect(frameOf({ ...CHUNK, frame: [] })).toBe('')
  })
})

describe('also_in trees', () => {
  // the score as the backend computes it (`passage.harmonic` of the two directions)
  const overlap = (contained: number, contains: number, alike: number) => ({ contained, contains, alike, score: (2 * contained * contains) / (contained + contains) })
  // a place as `/api/search/chunks` sends it, measured by words and by vectors
  const PLACE: Reference = {
    collection: 'A–E',
    document_id: 'a1',
    document: 'notes.md',
    seq: 3,
    header: 'Delivery',
    location: 'notes.md L4-5',
    line_start: 4,
    line_end: 5,
    score: 0.74,
    relation: 'contained',
    similarity: 0.97,
    to_parent: { words: overlap(0.97, 0.41, 0.38), embedding: overlap(0.9, 0.85, 0.88), chars: null },
    to_root: { words: overlap(0.5, 0.2, 0.2), embedding: null, chars: null },
    also_in: [],
  }

  test('the hint carries every measure, to the parent and to the match, and skips what is absent', () => {
    expect(overlapHint({ ...PLACE, to_parent: { ...PLACE.to_parent, chars: 0.5 } }).split('\n')).toEqual([
      'contained, query 0.74',
      'to parent',
      '  words: 0.58 (in 0.97, holds 0.41, alike 0.38)',
      '  embedding: 0.87 (in 0.90, holds 0.85, alike 0.88)',
      '  chars: 0.50',
      'to match',
      '  words: 0.29 (in 0.50, holds 0.20, alike 0.20)',
    ])
  })

  test('every place of a tree is counted once, depth first', () => {
    const tree = [{ ...PLACE, seq: 1, also_in: [{ ...PLACE, seq: 2, also_in: [{ ...PLACE, seq: 3 }] }] }, { ...PLACE, seq: 4 }]
    expect(everyPlace(tree).map((place) => ('seq' in place ? place.seq : 0))).toEqual([1, 2, 3, 4])
    expect(everyPlace([])).toEqual([])
  })
})

describe('excerpts', () => {
  const overlap = { contained: 1, contains: 1, alike: 1, score: 1 }
  // a place folded under a span, as `/api/search/excerpts` sends it
  const place = (document: string): Passage['also_in'][number] => ({
    collection: 'notes',
    document_id: document,
    document,
    seq_start: 2,
    seq_end: 2,
    header: 'Retries',
    location: `${document} L3-3`,
    line_start: 3,
    line_end: 3,
    score: 0.5,
    relation: 'duplicate',
    similarity: 1,
    to_parent: { words: overlap, embedding: null, chars: null },
    to_root: { words: overlap, embedding: null, chars: null },
    also_in: [],
  })
  const span = (also_in: Passage['also_in']): Excerpt['spans'][number] => ({
    header: 'Guide > Retries',
    location: 'guide.md L5-7',
    seq_start: 1,
    seq_end: 2,
    line_start: 5,
    line_end: 7,
    char_start: 40,
    char_end: 200,
    page_start: null,
    page_end: null,
    score: 1,
    also_in,
    aspects: [],
    aspect_scores: {},
  })
  const EXCERPT: Excerpt = {
    collection: 'notes',
    document_id: 'a1',
    document: 'guide.md',
    header: 'Guide',
    location: 'guide.md L5-20',
    seq_start: 1,
    seq_end: 6,
    line_start: 5,
    line_end: 20,
    char_start: 40,
    char_end: 900,
    page_start: null,
    page_end: null,
    text: 'One.\n\n[…]\n\nTwo.',
    score: 1,
    source_file: '/home/documents/guide.md',
    markdown_file: '/home/documents/guide.md.md',
    spans: [span([place('copy.md')]), span([]), span([place('guide.md'), place('other.md')])],
    aspects: [],
    aspect_scores: {},
  }

  test('an excerpt is told from a passage by its spans', () => {
    expect(isExcerpt(EXCERPT)).toBe(true)
  })

  test("an excerpt's folded places are its spans', in span order", () => {
    expect(alsoOf(EXCERPT).map((one) => one.document)).toEqual(['copy.md', 'guide.md', 'other.md'])
    expect(alsoDocuments(EXCERPT)).toBe(2)
  })

  test('its chunks run from the first span to the last', () => {
    expect(seqLabel(EXCERPT)).toBe('1–6')
  })
})

describe('questionLabels', () => {
  const asked = ['Why retry?', 'How long to wait?', 'When to stop?']
  const cases: Array<{ name: string; aspects: string[]; scores?: Record<string, number>; expected: Array<{ label: string; question: string; score?: string }> }> = [
    {
      name: 'each question by its place in what was asked, with its text',
      aspects: ['When to stop?', 'Why retry?'],
      expected: [
        { label: 'Q3', question: 'When to stop?' },
        { label: 'Q1', question: 'Why retry?' },
      ],
    },
    { name: 'none', aspects: [], expected: [] },
    {
      name: 'with the score the result matched it by, when the search said',
      aspects: ['Why retry?'],
      scores: { 'Why retry?': 0.8412 },
      expected: [{ label: 'Q1', question: 'Why retry?', score: '0.84' }],
    },
    { name: 'a question not asked this time names nothing', aspects: ['Old question?'], expected: [] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(questionLabels(one.aspects, asked, one.scores)).toEqual(one.expected)
    })
  }
})
