import { useEffect, useState } from 'react'
import { api, type Document, type Hit, type HotSection } from '../api'
import type { Anchor } from './anchor'
import { DocumentPanes } from './DocumentPanes'
import { Kv } from './Kv'
import { Mark } from './Mark'
import { CUT_REASONS, PIECE_NAMES, alsoCount, alsoOf, chunkSizes, cite, frameOf, headingOf, isHit, isSource, lastHeading, pieceMeta, piecesOf, position, seqLabel, type ChunkPiece, type Match, type Size } from './match'
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
export function MatchModal({ match, query, onClose }: { match: Match | null; query: string; onClose: () => void }) {
  return (
    <Modal open={match !== null} onClose={onClose} title={match?.document ?? ''} subtitle={match?.collection}>
      {/* Keyed by document: a result from another document starts its panels over, while one
          from the same document only scrolls them. */}
      {match !== null && <MatchBody key={match.document} match={match} query={query} />}
    </Modal>
  )
}

// A chunk or passage knows where it starts; a source only which heading its best chunk is under.
const anchorOf = (match: Match): Anchor => ({ heading: headingOf(match), offset: isSource(match) ? undefined : match.char_start })
// A section is a header; its last heading is the one the document is anchored by.
const sectionAnchor = (section: HotSection): Anchor => ({ heading: lastHeading(section.header) })

/** One edge of a chunk: the reason it was cut there, and what that reason means. */
function Cut({ side, reason }: { side: 'before' | 'after'; reason: Hit['start_reason'] }) {
  return (
    <div className="chunk-cut">
      cut {side}: <span className="chunk-cut-reason">{reason}</span> · {CUT_REASONS[reason]}
    </div>
  )
}

/** One piece of a chunk, outlined on hover with a hint naming its type, where it starts and how
 *  big it is. A heading is grey like the path above it: only a chunk of headings alone has any. */
function Piece({ piece, query }: { piece: ChunkPiece; query: string }) {
  return (
    <span className={piece.type === 'heading' ? 'chunk-piece chunk-heading' : 'chunk-piece'}>
      <Mark text={piece.text} query={query} />
      <span className="hint" role="tooltip">
        <strong>{PIECE_NAMES[piece.type]}</strong>
        <span className="sub mono">{pieceMeta(piece)}</span>
      </span>
    </span>
  )
}

/** A chunk exactly as the models read it: its frame (the heading path it was embedded after) on
 *  grey, then its own text on white piece by piece; the reason it was cut on each side, and how
 *  big each part is. */
function ChunkQuote({ hit, query }: { hit: Hit; query: string }) {
  const frame = frameOf(hit)
  return (
    <div className="chunk">
      <Cut side="before" reason={hit.start_reason} />
      <blockquote className="match-text chunk-text">
        {frame && (
          <span className="chunk-piece chunk-frame">
            {frame}
            <span className="hint" role="tooltip">
              <strong>Heading path</strong>
              <span className="sub mono">prepended to the chunk when it was embedded</span>
            </span>
          </span>
        )}
        {piecesOf(hit).map((piece) => (
          <Piece key={piece.position} piece={piece} query={query} />
        ))}
        <span className="match-seq" title={`chunk ${seqLabel(hit)}`}>{seqLabel(hit)}</span>
      </blockquote>
      <Cut side="after" reason={hit.end_reason} />
      <ChunkSizes hit={hit} />
    </div>
  )
}

/** The chunk's parts in characters, words and pieces, and the total against the chunk size
 *  its collection packs to now, which the heading path counts toward. */
function ChunkSizes({ hit }: { hit: Hit }) {
  const [limit, setLimit] = useState<number | null>(null)
  useEffect(() => {
    let live = true
    api
      .collection(hit.collection)
      .then((info) => {
        if (live) setLimit(info.effective.chunk_size)
      })
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [hit.collection])
  const sizes = chunkSizes(hit)
  const rows: [string, Size][] = [
    ['frame', sizes.frame],
    ['text', sizes.text],
    ['total', sizes.total],
  ]
  return (
    <table className="chunk-sizes">
      <thead>
        <tr>
          <th />
          <th>chars</th>
          <th>words</th>
          <th>pieces</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, size]) => (
          <tr key={name}>
            <th>{name}</th>
            <td>{name === 'total' && limit !== null ? `${size.chars} / ${limit}` : size.chars}</td>
            <td>{size.words}</td>
            <td>{size.pieces}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** The other places that say what the match says, folded into it by the search: how close each
 *  one is, where it sits, and how many there were in all when only the first few are listed. */
function AlsoIn({ match }: { match: Match }) {
  const count = alsoCount(match)
  if (count === 0) return null
  return (
    <div className="sections">
      <div className="sections-head">
        <span>Also in</span>
        <span className="mono muted">
          {count} {count === 1 ? 'place' : 'places'}
        </span>
      </div>
      {alsoOf(match).map((reference) => (
        <div key={`${reference.collection}:${reference.location}`} className="section-row">
          <span className="mono muted">{reference.similarity.toFixed(2)}</span>
          <span className="section-title">
            {reference.document} · {reference.header || '—'}
          </span>
          <span className="mono muted">{cite(reference.location, reference.document)}</span>
        </div>
      ))}
    </div>
  )
}

function MatchBody({ match, query }: { match: Match; query: string }) {
  const [tab, setTab] = useState(MATCH_TAB)
  // Fetched because the panes need the document's preview kind, which a match does not carry.
  const [row, setRow] = useState<Document | null>(null)
  // The section picked in a source is where the document opens.
  const [picked, setPicked] = useState<Anchor | null>(null)
  const name = match.document
  const source = isSource(match)

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
            ['Score', <span key="score" className="mono">{match.score.toFixed(2)}</span>],
            ['Collection', source ? match.collections.join(', ') : match.collection],
            source ? ['Chunks', match.chunks] : ['Position', position(match)],
            // a chunk shows its heading path once, on grey at the top of its quote
            ...(isHit(match) ? [] : [['Heading', headingOf(match) || '—'] as [string, string]]),
            ['Query', <span key="query" className="code">{query}</span>],
          ]}
        />
        {source ? (
          // The source as one block: a head row naming the document and its total, then its hot
          // sections indented under it, each citing itself without repeating the document name.
          <div className="sections">
            <div className="sections-head">
              <span>{match.document}</span>
              <span className="mono muted">{match.chunks} {match.chunks === 1 ? 'chunk' : 'chunks'}</span>
            </div>
            {match.sections.map((section) => (
              <div key={section.header} className="section-row" role="button" tabIndex={0} onClick={() => jump(section)}>
                <span className="mono muted">{section.score.toFixed(2)}</span>
                <span className="section-title">{section.header || '—'}</span>
                <span className="mono muted">
                  {section.chunks} {section.chunks === 1 ? 'chunk' : 'chunks'} · {cite(section.location, match.document)}
                </span>
              </div>
            ))}
          </div>
        ) : isHit(match) ? (
          <ChunkQuote hit={match} query={query} />
        ) : (
          <blockquote className="match-text">
            <Mark text={match.text} query={query} />
            <span className="match-seq" title={`chunk ${seqLabel(match)}`}>{seqLabel(match)}</span>
          </blockquote>
        )}
        <AlsoIn match={match} />
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        {/* The whole document, not the preview, streamed from the moment the modal opens so it
            is already at the match when its tab is chosen. */}
        {row !== null && <DocumentPanes doc={row.name} preview={row.preview} full anchor={picked ?? anchorOf(match)} shown={tab === DOCUMENT_TAB} />}
      </div>
    </>
  )
}
