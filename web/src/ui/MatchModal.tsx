import { Info } from 'lucide-react'
import { Fragment, useEffect, useState, type ReactNode } from 'react'
import { api, type Document, type Hit, type HotSection, type ScoreStep } from '../api'
import type { Anchor } from './anchor'
import { DocumentPanes } from './DocumentPanes'
import { Kv } from './Kv'
import { Mark } from './Mark'
import { MarkdownQuote } from './MarkdownQuote'
import { CUT_REASONS, PIECE_NAMES, RELATIONS, alsoOf, chunkSizes, cite, everyPlace, overlapHint, frameOf, headingOf, isHit, isSource, lastHeading, pieceMeta, piecesOf, position, seqLabel, type ChunkPiece, type Match, type Reference, type Size, isExcerpt, questionLabels } from './match'
import { errorText } from '../format'
import { Modal } from './Modal'
import { Tabs, type TabDef } from './Tabs'

const MATCH_TAB = 'modal-match'
const DOCUMENT_TAB = 'modal-document'
// A chunk or passage is one match; a source opens on the sections where its matches live.
const tabsFor = (source: boolean): TabDef[] => [
  { id: MATCH_TAB, label: source ? 'Sections' : 'Match' },
  { id: DOCUMENT_TAB, label: 'Document' },
]

/** What a search asked, when it asked several questions under one shared context. */
export interface Asked {
  questions: string[]
  context: string
}

/**
 * One search result, opened: the match itself, and the document it came from. Every page with a
 * `HitGrid` opens the same thing. An excerpt of a search of several questions shows the context
 * and the questions it answers where a single query shows the query.
 */
export function MatchModal({
  match,
  query,
  scoring = [],
  asked,
  onClose,
}: {
  match: Match | null
  query: string
  scoring?: ScoreStep[]
  asked?: Asked
  onClose: () => void
}) {
  return (
    <Modal open={match !== null} onClose={onClose} title={match?.document ?? ''} subtitle={match?.collection}>
      {/* Keyed by document: a result from another document starts its panels over, while one
          from the same document only scrolls them. */}
      {match !== null && <MatchBody key={match.document} match={match} query={query} scoring={scoring} asked={asked} />}
    </Modal>
  )
}

// A chunk or passage knows where it starts; a source only which heading its best chunk is under.
const anchorOf = (match: Match): Anchor => ({ heading: headingOf(match), offset: isSource(match) ? undefined : match.char_start })
// A section is a header; its last heading is the one the document is anchored by.
const sectionAnchor = (section: HotSection): Anchor => ({ heading: lastHeading(section.header) })

/** A result's score, with its lineage on hover or focus: each step that set or changed it, in
 *  the order they ran, and how. Nothing to hover when the search said nothing. */
function Score({ score, scoring }: { score: number; scoring: ScoreStep[] }) {
  return (
    <span className="score">
      <span className="mono">{score.toFixed(2)}</span>
      {scoring.length > 0 && (
        <span className="score-basis" tabIndex={0} aria-label="How the score is computed">
          <Info className="icon" />
          <span className="hint hint-below hint-wide" role="tooltip">
            <span className="label label-mono">Score lineage</span>
            <span className="score-lineage">
              {scoring.map((one) => (
                <Fragment key={`${one.step} ${one.rule}`}>
                  <span>{one.label}</span>
                  <span className="muted">{one.rule}</span>
                </Fragment>
              ))}
            </span>
          </span>
        </span>
      )}
    </span>
  )
}

/** One edge of a chunk: the reason it was cut there, and what that reason means. */
function Cut({ side, reason }: { side: 'before' | 'after'; reason: Hit['start_reason'] }) {
  return (
    <div className="chunk-cut">
      cut {side}: <span className="chunk-cut-reason">{reason}</span> · {CUT_REASONS[reason]}
    </div>
  )
}

/** One piece of a chunk, outlined on hover with a hint naming its type, where it starts and how
 *  big it is. */
function Piece({ piece, query }: { piece: ChunkPiece; query: string }) {
  return (
    <span className="chunk-piece">
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

/** What a row has read of its lines: nothing yet, a read on its way, the text, or why it failed. */
type Read = null | 'loading' | { text: string } | { error: string }

/** One place in `also_in`: a row naming it, which opens on click to the lines it points at,
 *  read from the document the first time rather than carried by every search result. */
function AlsoRow({ reference, query }: { reference: Reference; query: string }) {
  const [read, setRead] = useState<Read>(null)
  const opened = (open: boolean) => {
    if (!open || read !== null) return // one read per row, however often it is opened
    setRead('loading')
    api
      .lines(reference.document, reference.line_start, reference.line_end)
      .then((found) => setRead({ text: found.text }))
      .catch((cause: unknown) => setRead({ error: errorText(cause) }))
  }
  return (
    <details onToggle={(event) => opened(event.currentTarget.open)}>
      <summary className="section-row">
        <span className="mono muted" title={overlapHint(reference)}>
          {RELATIONS[reference.relation]} {reference.similarity.toFixed(2)}
        </span>
        <span className="section-title">
          {reference.document} · {reference.header || '—'}
        </span>
        <span className="mono muted">{cite(reference.location, reference.document)}</span>
      </summary>
      <blockquote className="match-text also-text">
        {read === null || read === 'loading' ? 'Loading…' : 'error' in read ? read.error : <Mark text={read.text} query={query} />}
      </blockquote>
    </details>
  )
}

/** A place and, indented under it, the places folded into it: the tree the search built. Keyed
 *  by position, since two places may cite the same lines and the order is fixed. */
function AlsoPlace({ reference, query }: { reference: Reference; query: string }) {
  return (
    <>
      <AlsoRow reference={reference} query={query} />
      {reference.also_in.length > 0 && (
        <div className="also-nested">
          {reference.also_in.map((child, at) => (
            <AlsoPlace key={at} reference={child} query={query} />
          ))}
        </div>
      )}
    </>
  )
}

/** The other places that say what the match says, folded into it by the search: how close each
 *  one is, where it sits, what it repeats, and how many there are at every level. */
function AlsoIn({ match, query }: { match: Match; query: string }) {
  const places = alsoOf(match)
  const count = everyPlace(places).length
  if (count === 0) return null
  return (
    <div className="sections">
      <div className="sections-head">
        <span>Also in</span>
        <span className="mono muted">
          {count} {count === 1 ? 'place' : 'places'}
        </span>
      </div>
      {places.map((reference, at) => (
        <AlsoPlace key={at} reference={reference} query={query} />
      ))}
    </div>
  )
}

/** The rows that say what was asked: the context and each question the excerpt answers, labelled
 *  as the results are ("Q2"), when several were asked; else the query. */
function askedRows(match: Match, query: string, asked?: Asked): [string, ReactNode][] {
  if (asked === undefined || asked.questions.length < 2 || !isExcerpt(match)) {
    return [['Query', <span key="query" className="code">{query}</span>]]
  }
  return [
    ...(asked.context === '' ? [] : [['Context', asked.context] as [string, ReactNode]]),
    ...questionLabels(match.aspects, asked.questions, match.aspect_scores).map(({ label, question, score }): [string, ReactNode] => [
      label,
      <span key={label}>
        <span className="code">{question}</span>
        {score !== undefined && <span className="mono muted"> · {score}</span>}
      </span>,
    ]),
  ]
}

function MatchBody({ match, query, scoring, asked }: { match: Match; query: string; scoring: ScoreStep[]; asked?: Asked }) {
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
            ['Score', <Score key="score" score={match.score} scoring={scoring} />],
            ['Collection', source ? match.collections.join(', ') : match.collection],
            source ? ['Chunks', match.chunks] : ['Position', position(match)],
            // a chunk shows its heading path once, on grey at the top of its quote
            ...(isHit(match) ? [] : [['Heading', headingOf(match) || '—'] as [string, string]]),
            ...askedRows(match, query, asked),
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
          <blockquote className="match-text match-markdown">
            <MarkdownQuote text={match.text} query={query} />
            <span className="match-seq" title={`chunk ${seqLabel(match)}`}>{seqLabel(match)}</span>
          </blockquote>
        )}
        <AlsoIn match={match} query={query} />
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        {/* The whole document, not the preview, streamed from the moment the modal opens so it
            is already at the match when its tab is chosen. */}
        {row !== null && <DocumentPanes doc={row.name} preview={row.preview} full anchor={picked ?? anchorOf(match)} shown={tab === DOCUMENT_TAB} />}
      </div>
    </>
  )
}
