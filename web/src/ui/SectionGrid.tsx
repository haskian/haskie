import type { MappedDocument, MappedSection, RelatedSection } from '../api'
import { href } from '../router'
import { cite, fillOf, plural, scoreStyle } from './match'

// Where a related section is, as far as it differs from its pick: the same section of the same
// document, chunked by another collection, names that collection.
const placeOf = (near: RelatedSection, pick: MappedSection): string =>
  [near.document !== pick.document && near.document, near.collection !== pick.collection && near.collection].filter(Boolean).join(' · ')

/** A section of the map, or one listed under it, opened: its document's sections, at this one,
 *  with what the map said about it. */
export type OpenedSection = MappedSection | RelatedSection

const isPick = (section: OpenedSection): section is MappedSection => 'related' in section

/** The map of sections a topic touches: a tile each, naming the section and what it is about.
 *  Under it, the sections it covers best, which the map did not pick again. A tile, or a section
 *  under it, opens its document's sections. */
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
            <Descriptors words={section.descriptors} />
            <footer className="hit-foot">
              <span>{cite(section.location, section.document) || ' '}</span>
              <span>
                {plural(section.chars, 'char')} · {plural(section.chunks, 'matched chunk')}
              </span>
            </footer>
            <Related pick={section} onOpen={onOpen} />
          </div>
        </article>
      ))}
    </div>
  )
}

/** The sections a pick covers best, which the map did not pick again, each with how closely the
 *  pick covers it; nothing when there are none. A row opens that section. */
function Related({ pick, onOpen }: { pick: MappedSection; onOpen?: (section: OpenedSection) => void }) {
  const related = pick.related ?? []
  if (related.length === 0) return null
  return (
    <ul className="related" aria-label="Related sections">
      {related.map((near) => (
        <li
          key={`${near.collection}:${near.document_id}:${near.line_start}`}
          onClick={(event) => {
            event.stopPropagation()
            onOpen?.(near)
          }}
        >
          <span className="related-title">{near.header || near.document}</span>
          {placeOf(near, pick) && <span className="related-place mono muted">{placeOf(near, pick)}</span>}
          <span className="mono muted">{near.similarity.toFixed(2)}</span>
        </li>
      ))}
    </ul>
  )
}

/** What the map said about the opened section, under its row in the document's sections: for a
 *  pick, its score, how many of its chunks matched and the sections it covers; for a section
 *  listed under a pick, its score and how closely that pick covers it. */
export function OpenedDetail({ section, onOpen }: { section: OpenedSection; onOpen?: (section: OpenedSection) => void }) {
  return (
    <div className="opened-detail">
      <span className="mono">
        {isPick(section)
          ? `score ${section.score.toFixed(2)} · ${plural(section.chunks, 'matched chunk')}`
          : `score ${section.score.toFixed(2)} · similarity ${section.similarity.toFixed(2)} to its pick`}
      </span>
      {isPick(section) && <Related pick={section} onOpen={onOpen} />}
    </div>
  )
}

/** What a section is about, one chip per descriptor; nothing when it has none. */
export function Descriptors({ words }: { words: string[] }) {
  if (words.length === 0) return null
  return (
    <div className="descriptors">
      {words.map((word) => (
        <span key={word} className="descriptor">
          {word}
        </span>
      ))}
    </div>
  )
}

/** The documents the search reached hardest, best first: what each is about, how much of it
 *  matched, how many of the map's sections are in it and which collections hold it; nothing when
 *  there are none. A row opens the document. */
export function MapDocuments({ documents }: { documents: MappedDocument[] }) {
  if (documents.length === 0) return null
  return (
    <div className="sections" aria-label="Documents">
      <div className="sections-head">
        <span>Documents</span>
        <span className="mono muted">{plural(documents.length, 'document')}</span>
      </div>
      {documents.map((one) => (
        <a key={one.document_id} className="section-row" href={href({ name: 'documents', document: one.document })}>
          <span className="mono muted">{one.score.toFixed(2)}</span>
          <span className="section-title">
            {one.document}
            {one.description && <span className="muted"> · {one.description}</span>}
          </span>
          <span className="mono muted">
            {plural(one.chunks, 'chunk')} · {plural(one.sections, 'section')} · {one.collections.join(', ')}
          </span>
        </a>
      ))}
    </div>
  )
}
