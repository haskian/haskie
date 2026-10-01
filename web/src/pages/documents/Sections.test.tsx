import { describe, expect, test } from 'bun:test'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { Sections } from '../../api'
import { SectionsTab } from './Sections'

type Section = Sections['sections'][number]

// What `GET /api/documents/{name}/sections` answers for a PDF: the whole document, a chapter and
// a section inside it, described by c-TF-IDF.
const WHOLE: Section = {
  id: '3mJr7AoUXx2Wqd',
  parent_id: null,
  headings: [],
  seq_start: 1,
  seq_end: 40,
  line_start: 1,
  line_end: 900,
  char_start: 0,
  char_end: 52_000,
  byte_start: 0,
  byte_end: 52_340,
  page_start: 1,
  page_end: 30,
  descriptors: ['aggregates', 'bounded contexts'],
}
const CHAPTER: Section = {
  ...WHOLE,
  id: '9xKqP2rWn4Vb',
  parent_id: WHOLE.id,
  headings: ['Aggregates'],
  seq_start: 3,
  seq_end: 20,
  line_start: 40,
  line_end: 410,
  page_start: 3,
  page_end: 12,
  descriptors: ['consistency boundary', 'invariants'],
}
const RULE: Section = {
  ...CHAPTER,
  id: 'Fh7tY1mZs8Lc',
  parent_id: CHAPTER.id,
  headings: ['Aggregates', 'Rule: Design Small Aggregates'],
  line_start: 120,
  line_end: 180,
  page_start: 5,
  page_end: 5,
  descriptors: [],
}
const FOUND: Sections = { sections: [WHOLE, CHAPTER, RULE], described_by: 'c-tf-idf' }

const cases: Array<{ name: string; element: ReactElement; contains: string[]; missing?: string[] }> = [
  {
    name: 'each section by its heading, indented by its depth, with its descriptors as tags',
    element: <SectionsTab found={FOUND} />,
    contains: [
      '<span>Descriptors by c-tf-idf.</span></p>',
      '<li style="--depth:0"><span class="section-toc-head"><span class="section-heading">The whole document</span><span class="mono muted">p. 1–30</span></span><div class="descriptors"><span class="descriptor">aggregates</span>',
      '<li style="--depth:0"><span class="section-toc-head"><span class="section-heading">Aggregates</span>',
      '<li style="--depth:1"><span class="section-toc-head"><span class="section-heading">Rule: Design Small Aggregates</span><span class="mono muted">p. 5</span></span></li>',
    ],
  },
  {
    name: 'nothing cached yet: says so, and names no strategy',
    element: <SectionsTab found={{ sections: [], described_by: null }} />,
    contains: ['<p class="info"><svg', '<span>No sections yet'],
    missing: ['<ol', 'Descriptors by'],
  },
  {
    name: 'a document without pages: its lines',
    element: <SectionsTab found={{ sections: [{ ...CHAPTER, page_start: null, page_end: null }], described_by: 'llm' }} />,
    contains: ['<span class="mono muted">lines 40–410</span>'],
  },
  {
    name: 'not yet described: the sections, without a strategy line',
    element: <SectionsTab found={{ sections: [CHAPTER], described_by: null }} />,
    contains: ['class="list section-toc"'],
    missing: ['Descriptors by'],
  },
]

describe('SectionsTab', () => {
  for (const one of cases) {
    test(one.name, () => {
      const html = renderToStaticMarkup(one.element)
      for (const needle of one.contains) expect(html).toContain(needle)
      for (const needle of one.missing ?? []) expect(html).not.toContain(needle)
    })
  }
})
