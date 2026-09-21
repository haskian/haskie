import { useEffect, useState } from 'react'
import { api, type Document, type Hit } from '../api'
import type { Anchor } from './anchor'
import { DocumentPanes } from './DocumentPanes'
import { HitGrid } from './HitGrid'
import { Kv } from './Kv'
import { Mark } from './Mark'
import { isHit, position, type Match } from './match'
import { Modal } from './Modal'
import { Tabs, type TabDef } from './Tabs'

const MATCH_TAB = 'modal-match'
const DOCUMENT_TAB = 'modal-document'
// A passage is one match; a document match opens on every passage behind it.
const tabsFor = (single: boolean): TabDef[] => [
  { id: MATCH_TAB, label: single ? 'Match' : 'Matches' },
  { id: DOCUMENT_TAB, label: 'Document' },
]

/**
 * One search result, opened: the match itself, and the document it came from. Every page with a
 * `HitGrid` opens the same thing. `collections` is the scope the query ran over, so a document
 * match can fetch its passages from the same scan.
 */
export function MatchModal({
  hit,
  query,
  collections,
  onClose,
}: {
  hit: Match | null
  query: string
  collections?: string[]
  onClose: () => void
}) {
  return (
    <Modal open={hit !== null} onClose={onClose} title={hit?.doc ?? ''} subtitle={hit?.collection}>
      {/* Keyed by document: a result from another document starts its panels over, while one
          from the same document only scrolls them. */}
      {hit !== null && <MatchBody key={hit.doc} hit={hit} query={query} collections={collections} />}
    </Modal>
  )
}

// A passage knows where it starts; a document match only which heading it is under.
const anchorOf = (hit: Match): Anchor => ({ heading: hit.heading, offset: isHit(hit) ? hit.char_start : undefined })

function MatchBody({ hit, query, collections }: { hit: Match; query: string; collections?: string[] }) {
  const [tab, setTab] = useState(MATCH_TAB)
  // Fetched because the panes need the document's preview kind, which a hit does not carry.
  const [row, setRow] = useState<Document | null>(null)
  // A document match lists every passage behind it; the one picked is where the document opens.
  const [passages, setPassages] = useState<Hit[] | null>(null)
  const [picked, setPicked] = useState<Hit | null>(null)
  const name = hit.doc
  const single = isHit(hit)

  useEffect(() => {
    let live = true
    api
      .document(name)
      .then((fetched) => {
        if (live) setRow(fetched)
      })
      .catch(() => undefined)
    if (!single) {
      api
        .documentPassages(name, query, collections)
        .then((found) => {
          if (live) setPassages(found)
        })
        .catch(() => undefined)
    }
    return () => {
      live = false
    }
  }, [name, query, collections, single])

  const jump = (passage: Hit) => {
    setPicked(passage)
    setTab(DOCUMENT_TAB)
  }

  return (
    <>
      <Tabs tabs={tabsFor(single)} selected={tab} onSelect={setTab} />
      <div id={MATCH_TAB} role="tabpanel" className="match" hidden={tab !== MATCH_TAB}>
        <Kv
          rows={[
            ['Score', <span key="score" className="mono">{hit.score.toFixed(2)}</span>],
            ['Collection', hit.collection],
            single ? ['Position', position(hit)] : ['Passages', hit.chunks],
            ['Heading', hit.heading || '—'],
            ['Query', <span key="query" className="code">{query}</span>],
          ]}
        />
        {single ? (
          <blockquote className="match-text">
            <Mark text={hit.text} query={query} />
          </blockquote>
        ) : (
          passages !== null && <HitGrid hits={passages} query={query} onOpen={jump} />
        )}
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        {/* The whole document, not the preview, streamed from the moment the modal opens so it
            is already at the match when its tab is chosen. */}
        {row !== null && (
          <DocumentPanes doc={row.name} preview={row.preview} full anchor={anchorOf(picked ?? hit)} shown={tab === DOCUMENT_TAB} />
        )}
      </div>
    </>
  )
}
