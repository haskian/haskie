import type { CSSProperties } from 'react'
import type { Sections } from '../../api'
import { Descriptors, Info, pagesOf, span, WHOLE_DOCUMENT } from '../../ui'

/** A document's table of contents, each section indented by its depth, with what it is about
 *  as tags. The first section is the whole document, under no heading. */
export function SectionsTab({ found }: { found: Sections }) {
  if (found.sections.length === 0) return <Info>No sections yet: they are cut when the document is embedded.</Info>
  return (
    <div className="section">
      {found.described_by !== null && <Info>Descriptors by {found.described_by}.</Info>}
      <ol className="list section-toc">
        {found.sections.map((section) => (
          <li key={section.id} style={{ '--depth': Math.max(0, section.headings.length - 1) } as CSSProperties}>
            <span className="section-toc-head">
              <span className="section-heading">{section.headings.at(-1) ?? WHOLE_DOCUMENT}</span>
              <span className="mono muted">{pagesOf(section.page_start, section.page_end) || span('line', section.line_start, section.line_end)}</span>
            </span>
            <Descriptors words={section.descriptors} />
          </li>
        ))}
      </ol>
    </div>
  )
}
