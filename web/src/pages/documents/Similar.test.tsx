import { describe, expect, test } from 'bun:test'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { Similar } from '../../api'
import { Duplicates, SimilarDocuments } from './Similar'

// What `GET /api/documents/{name}/similar` answers for a second edition imported beside the first.
const SIMILAR: Similar = {
  identical: ['ddd-copy.pdf', 'ddd.pdf'],
  nearest: [
    { document: 'ddd.pdf', similarity: 1 },
    { document: 'implementing-ddd.epub', similarity: 0.8731 },
  ],
}

const cases: Array<{ name: string; element: ReactElement; contains: string[]; missing?: string[] }> = [
  {
    name: 'the same file under other names, each a link, then the nearest with their similarity',
    element: <SimilarDocuments similar={SIMILAR} />,
    contains: [
      '<p class="notice">The same file is already imported as <a href="#/documents/ddd-copy.pdf">ddd-copy.pdf</a>, <a href="#/documents/ddd.pdf">ddd.pdf</a>.</p>',
      '<a href="#/documents/implementing-ddd.epub">implementing-ddd.epub</a><span class="sub">similarity 0.87</span>',
      '<span class="sub">similarity 1.00</span>',
    ],
  },
  {
    name: 'no copy and nothing embedded to compare with: says so, and no notice',
    element: <SimilarDocuments similar={{ identical: [], nearest: [] }} />,
    contains: ['<p class="muted">Nothing to compare with'],
    missing: ['notice', '<ul'],
  },
  {
    name: 'at staging, the copy comes with the advice not to import it',
    element: <Duplicates names={['ddd.pdf']} advice="Importing it again only adds a copy." />,
    contains: ['<a href="#/documents/ddd.pdf">ddd.pdf</a>. Importing it again only adds a copy.</p>'],
  },
]

describe('SimilarDocuments', () => {
  for (const one of cases) {
    test(one.name, () => {
      const html = renderToStaticMarkup(one.element)
      for (const needle of one.contains) expect(html).toContain(needle)
      for (const needle of one.missing ?? []) expect(html).not.toContain(needle)
    })
  }
})
