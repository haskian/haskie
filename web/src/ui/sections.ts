import type { MappedSection, OutlineSection, RelatedSection } from '../api'

/** Where a section of the map or of an outline sits in one document. */
export type Spot = Pick<MappedSection | RelatedSection | OutlineSection, 'header' | 'line_start' | 'line_end'>

/** A section's keywords as the map shows them: the ones that set it apart from the other sections
 *  on the map first, marked, then the rest of what it is about. */
export function keywordsOf(section: Pick<MappedSection, 'keywords' | 'distinct'>): Array<{ word: string; distinct: boolean }> {
  const distinct = new Set(section.distinct)
  return [
    ...section.distinct.map((word) => ({ word, distinct: true })),
    ...section.keywords.filter((word) => !distinct.has(word)).map((word) => ({ word, distinct: false })),
  ]
}

/** The outline entry a section of the map is: the one under the same heading path whose lines
 *  overlap its own most. The outline was cut from one chunking and the map from its collection's,
 *  so the two can bound a section a few lines apart. -1 when none overlaps. */
export function outlineIndex(outline: Spot[], section: Spot): number {
  let best = -1
  let most = 0
  outline.forEach((entry, at) => {
    if (entry.header !== section.header) return
    const shared = Math.min(entry.line_end, section.line_end) - Math.max(entry.line_start, section.line_start) + 1
    if (shared > most) {
      best = at
      most = shared
    }
  })
  return best
}
