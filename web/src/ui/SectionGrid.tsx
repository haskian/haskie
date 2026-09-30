import type { CSSProperties } from 'react'
import type { MappedSection, RelatedSection } from '../api'
import { cite, fillOf } from './match'
import { keywordsOf } from './sections'

// `--score` drives the bar under a tile, as in `HitGrid`: its place among the others.
const scoreStyle = (fill: number): CSSProperties => ({ '--score': fill }) as CSSProperties
const count = (n: number, unit: string): string => `${n} ${unit}${n === 1 ? '' : 's'}`
// Where a related section is, as far as it differs from its pick: the same section of the same
// document, chunked by another collection, names that collection.
const placeOf = (near: RelatedSection, pick: MappedSection): string =>
  [near.document !== pick.document && near.document, near.collection !== pick.collection && near.collection].filter(Boolean).join(' · ')

/** A section of the map, or one listed under it, opened: its document's outline, at the section. */
export type OpenedSection = Pick<MappedSection | RelatedSection, 'document' | 'header' | 'line_start' | 'line_end'>

/** The map of sections a topic touches: a tile each, naming the section and what it is about,
 *  the words that set it apart marked first. Under it, the sections it covers best, which the
 *  map did not pick again. A tile, or a section under it, opens its document's outline. */
export function SectionGrid({ sections, onOpen }: { sections: MappedSection[]; onOpen?: (section: OpenedSection) => void }) {
  const scores = sections.map((section) => section.score)
  const best = Math.max(...scores)
  const worst = Math.min(...scores)
  return (
    <div className="hits">
      {sections.map((section) => (
        <article
          key={`${section.collection}:${section.document_id}:${section.seq_start}`}
          className="hit"
          style={scoreStyle(fillOf(section.score, best, worst))}
          onClick={() => onOpen?.(section)}
        >
          <header className="hit-head">
            <span className="tag">
              <span className="kind">{section.collection}</span>
              <span>{section.document}</span>
            </span>
            <span className="mono muted">{section.score.toFixed(2)}</span>
          </header>
          <div className="hit-body">
            <p className="section-heading">{section.header || 'The whole document'}</p>
            {section.keywords.length > 0 && (
              <div className="keywords">
                {keywordsOf(section).map(({ word, distinct }) => (
                  <span key={word} className={distinct ? 'keyword distinct' : 'keyword'}>
                    {word}
                  </span>
                ))}
              </div>
            )}
            <footer className="hit-foot">
              <span>{cite(section.location, section.document) || ' '}</span>
              <span>
                {count(section.chars, 'char')} · {count(section.chunks, 'matched chunk')}
              </span>
            </footer>
            {(section.related ?? []).length > 0 && (
              <ul className="related" aria-label="Related sections">
                {(section.related ?? []).map((near) => (
                  <li
                    key={`${near.collection}:${near.document_id}:${near.line_start}`}
                    onClick={(event) => {
                      event.stopPropagation()
                      onOpen?.(near)
                    }}
                  >
                    <span className="related-title">{near.header || near.document}</span>
                    {placeOf(near, section) && <span className="related-place mono muted">{placeOf(near, section)}</span>}
                    <span className="mono muted">{near.similarity.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </article>
      ))}
    </div>
  )
}
