import { describe, expect, test } from 'bun:test'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { CollectionsTab } from './Collections'

// What `GET /api/collections` names, and the two of them `GET /api/documents/{name}/collections`
// says hold the document.
const ALL = ['Distributed-Systems', 'Golang', 'Software-Architecture']
const HELD = ['Distributed-Systems', 'Software-Architecture']
const noop = () => undefined
const tab = (props: Partial<Parameters<typeof CollectionsTab>[0]>): ReactElement => (
  <CollectionsTab held={HELD} all={ALL} attachable busy={false} onAttach={noop} onDetach={noop} {...props} />
)

const ADD = '<button class="btn btn-ghost" type="button" aria-label="Add">'
const ADD_DISABLED = '<button class="btn btn-ghost" type="button" aria-label="Add" disabled="">'
const REMOVE = '<button class="btn btn-ghost" type="button" aria-label="Remove">'

const cases: Array<{ name: string; element: ReactElement; contains: string[]; missing?: string[] }> = [
  {
    name: 'held ones link to their collection with a remove; the rest are available with an add',
    element: tab({}),
    contains: [
      'In collections · 2',
      '<a href="#/collections/Distributed-Systems">Distributed-Systems</a>',
      REMOVE,
      'Available · 1',
      `<span class="list-text">Golang</span>${ADD}`,
    ],
    missing: ['role="status"'],
  },
  {
    name: 'in no collection: every one is available',
    element: tab({ held: [] }),
    contains: ['In collections · 0', 'Available · 3'],
    missing: [REMOVE],
  },
  {
    name: 'in every collection: none is available',
    element: tab({ held: ALL }),
    contains: ['In collections · 3', 'Available · 0'],
    missing: ['aria-label="Add"'],
  },
  {
    name: 'not imported yet: says so, and add is disabled',
    element: tab({ attachable: false }),
    contains: ['once it is imported', ADD_DISABLED, REMOVE],
  },
  {
    name: 'busy: both buttons are disabled',
    element: tab({ busy: true }),
    contains: [ADD_DISABLED, 'aria-label="Remove" disabled=""'],
  },
]

describe('CollectionsTab', () => {
  for (const one of cases) {
    test(one.name, () => {
      const html = renderToStaticMarkup(one.element)
      for (const needle of one.contains) expect(html).toContain(needle)
      for (const needle of one.missing ?? []) expect(html).not.toContain(needle)
    })
  }
})
