import type { CSSProperties } from 'react'
import { Mark } from './Mark'
import { fillOf, headingOf, isSource, position, type Match } from './match'

// `--score` drives the bar under a tile: the result's place among the others, not its raw score.
const scoreStyle = (fill: number): CSSProperties => ({ '--score': fill }) as CSSProperties

// A source shows what the document is about when it has a description; a chunk or passage shows
// the text that matched. The footer's right cell is the position, or the size of the evidence.
const textOf = (match: Match): string => (isSource(match) ? match.description || match.text : match.text)
const metaOf = (match: Match): string => (isSource(match) ? `${match.chunks} chunks · ${match.sections.length} sections` : position(match))
const keyOf = (match: Match): string =>
  isSource(match) ? `${match.collection}:${match.document}` : `${match.collection}:${match.document}:${match.char_start}` // offsets are unique in a document

/** The result grid, in any of its shapes: chunks, passages or excerpts that matched, or sources. */
export function HitGrid<T extends Match>({ results, query, onOpen }: { results: T[]; query: string; onOpen?: (match: T) => void }) {
  const scores = results.map((match) => match.score)
  const best = Math.max(...scores)
  const worst = Math.min(...scores)
  return (
    <div className="hits">
      {results.map((match) => (
        <article key={keyOf(match)} className="hit" style={scoreStyle(fillOf(match.score, best, worst))} onClick={() => onOpen?.(match)}>
          <header className="hit-head">
            <span className="tag">
              <span className="kind">{match.collection}</span>
              <span>{match.document}</span>
            </span>
            <span className="mono muted">{match.score.toFixed(2)}</span>
          </header>
          <div className="hit-body">
            <p className="hit-text">
              <Mark text={textOf(match)} query={query} />
            </p>
            <footer className="hit-foot">
              <span>{headingOf(match) || (isSource(match) ? '' : match.header)}</span>
              <span>{metaOf(match)}</span>
            </footer>
          </div>
        </article>
      ))}
    </div>
  )
}
