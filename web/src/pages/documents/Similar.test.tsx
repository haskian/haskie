import { describe, expect, test } from 'bun:test'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { Similar } from '../../api'
import { Duplicate, SimilarDocuments } from './Similar'

// What `GET /api/documents/{name}/similar` answers for a second edition imported beside the first.
// The same file is never two documents, so there is no copy to list.
const SIMILAR: Similar = {
  nearest: [
    { document: 'ddd.pdf', similarity: 1 },
    { document: 'implementing-ddd.epub', similarity: 0.8731 },
  ],
}

const cases: Array<{ name: string; element: ReactElement; contains: string[]; missing?: string[] }> = [
  {
    name: 'the nearest, each a link with its similarity, and no notice',
    element: <SimilarDocuments similar={SIMILAR} />,
    missing: ['notice'],
    contains: [
      '<a href="#/documents/implementing-ddd.epub">implementing-ddd.epub</a><span class="sub">similarity 0.87</span>',
      '<span class="sub">similarity 1.00</span>',
    ],
  },
  {
    name: 'nothing embedded to compare with: says so',
    element: <SimilarDocuments similar={{ nearest: [] }} />,
    contains: ['<div class="modal-message modal-message-info" role="status"><svg', '<span>Nothing to compare with'],
    missing: ['notice', '<ul'],
  },
  {
    name: 'at staging, the same file names the document it already is, and is not imported',
    element: <Duplicate name="ddd.pdf" />,
    contains: ['<a href="#/documents/ddd.pdf">ddd.pdf</a>. It is not imported again.</span></div>'],
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
