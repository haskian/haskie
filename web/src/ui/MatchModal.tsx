import { useEffect, useState } from 'react'
import { api, type Document, type HotSection } from '../api'
import type { Anchor } from './anchor'
import { DocumentPanes } from './DocumentPanes'
import { Kv } from './Kv'
import { Mark } from './Mark'
import { headingOf, isSource, position, type Match } from './match'
import { Modal } from './Modal'
import { Tabs, type TabDef } from './Tabs'

const MATCH_TAB = 'modal-match'
const DOCUMENT_TAB = 'modal-document'
// A chunk or passage is one match; a source opens on the sections where its matches live.
const tabsFor = (source: boolean): TabDef[] => [
  { id: MATCH_TAB, label: source ? 'Sections' : 'Match' },
  { id: DOCUMENT_TAB, label: 'Document' },
]

/**
 * One search result, opened: the match itself, and the document it came from. Every page with a
 * `HitGrid` opens the same thing.
 */
export function MatchModal({ hit, query, onClose }: { hit: Match | null; query: string; onClose: () => void }) {
  return (
    <Modal open={hit !== null} onClose={onClose} title={hit?.doc ?? ''} subtitle={hit?.collection}>
      {/* Keyed by document: a result from another document starts its panels over, while one
          from the same document only scrolls them. */}
      {hit !== null && <MatchBody key={hit.doc} hit={hit} query={query} />}
    </Modal>
  )
}

// A chunk or passage knows where it starts; a source only which heading its best chunk is under.
const anchorOf = (hit: Match): Anchor => ({ heading: headingOf(hit), offset: isSource(hit) ? undefined : hit.char_start })
// A section is a heading breadcrumb; its last step is the heading the document is anchored by.
const sectionAnchor = (section: HotSection): Anchor => ({ heading: section.header.split(' > ').at(-1) ?? '' })

function MatchBody({ hit, query }: { hit: Match; query: string }) {
  const [tab, setTab] = useState(MATCH_TAB)
  // Fetched because the panes need the document's preview kind, which a hit does not carry.
  const [row, setRow] = useState<Document | null>(null)
  // The section picked in a source is where the document opens.
  const [picked, setPicked] = useState<Anchor | null>(null)
  const name = hit.doc
  const source = isSource(hit)

  useEffect(() => {
    let live = true
    api
      .document(name)
      .then((fetched) => {
        if (live) setRow(fetched)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [name])

  const jump = (section: HotSection) => {
    setPicked(sectionAnchor(section))
    setTab(DOCUMENT_TAB)
  }

  return (
    <>
      <Tabs tabs={tabsFor(source)} selected={tab} onSelect={setTab} />
      <div id={MATCH_TAB} role="tabpanel" className="match" hidden={tab !== MATCH_TAB}>
        <Kv
          rows={[
            ['Score', <span key="score" className="mono">{hit.score.toFixed(2)}</span>],
            ['Collection', source ? hit.collections.join(', ') : hit.collection],
            source ? ['Chunks', hit.chunks] : ['Position', position(hit)],
            ['Heading', headingOf(hit) || '—'],
            ['Query', <span key="query" className="code">{query}</span>],
          ]}
        />
        {source ? (
          <ul className="list">
            {hit.sections.map((section) => (
              <li key={section.header} className="list-item" role="button" tabIndex={0} onClick={() => jump(section)}>
                <span className="mono muted">{section.score.toFixed(2)}</span>
                <span>{section.header || '—'}</span>
                <span className="mono muted">
                  {section.chunks} {section.chunks === 1 ? 'chunk' : 'chunks'} · {section.location}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <blockquote className="match-text">
            <Mark text={hit.text} query={query} />
          </blockquote>
        )}
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        {/* The whole document, not the preview, streamed from the moment the modal opens so it
            is already at the match when its tab is chosen. */}
        {row !== null && <DocumentPanes doc={row.name} preview={row.preview} full anchor={picked ?? anchorOf(hit)} shown={tab === DOCUMENT_TAB} />}
      </div>
    </>
  )
}
