import type { CSSProperties, ReactNode } from 'react'
import type { DocumentMatch, Hit } from '../api'
import { Mark } from './Mark'
import { fillOf } from './match'

// `--score` drives the bar under a tile: the result's place among the others, not its raw score.
const scoreStyle = (fill: number): CSSProperties => ({ '--score': fill }) as CSSProperties

type HitGridProps =
  | { hits: Hit[]; query: string; onOpen?: (hit: Hit) => void }
  | { matches: DocumentMatch[]; query: string; onOpen?: (match: DocumentMatch) => void }

// Both shapes render the same card; only the key, the marked text and the footer's right cell
// differ, so each is read into one row and the grid is written once.
interface HitRow {
  key: string
  collection: string
  doc: string
  score: number
  text: string
  heading: string
  meta: ReactNode
  open: () => void
}

/** The result grid, in either of its two shapes: passages that matched, or documents that did. */
export function HitGrid(props: HitGridProps) {
  const rows: HitRow[] =
    'hits' in props
      ? props.hits.map((hit) => ({
          key: `${hit.collection}:${hit.doc}:${hit.char_start}`, // chunk ids restart per part; the offset is unique in a document
          collection: hit.collection,
          doc: hit.doc,
          score: hit.score,
          text: hit.text,
          heading: hit.heading || hit.header,
          meta: (
            <>
              {hit.page_start !== null && `p. ${hit.page_start} · `}chunk {hit.chunk_id}
            </>
          ),
          open: () => props.onOpen?.(hit),
        }))
      : props.matches.map((match) => ({
          key: `${match.collection}:${match.doc}`,
          collection: match.collection,
          doc: match.doc,
          score: match.score,
          text: match.description || match.text,
          heading: match.heading,
          meta: `${match.chunks} chunks`,
          open: () => props.onOpen?.(match),
        }))

  const scores = rows.map((row) => row.score)
  const best = Math.max(...scores)
  const worst = Math.min(...scores)
  return (
    <div className="hits">
      {rows.map((row) => (
        <article key={row.key} className="hit" style={scoreStyle(fillOf(row.score, best, worst))} onClick={row.open}>
          <header className="hit-head">
            <span className="tag">
              <span className="kind">{row.collection}</span>
              <span>{row.doc}</span>
            </span>
            <span className="mono muted">{row.score.toFixed(2)}</span>
          </header>
          <div className="hit-body">
            <p className="hit-text">
              <Mark text={row.text} query={props.query} />
            </p>
            <footer className="hit-foot">
              <span>{row.heading}</span>
              <span>{row.meta}</span>
            </footer>
          </div>
        </article>
      ))}
    </div>
  )
}
