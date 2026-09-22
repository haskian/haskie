import type { Hit, Passage, Source } from '../api'

/** Any shape a search result arrives in: a chunk, a passage (or excerpt), or a document. */
export type Match = Hit | Passage | Source

/** A chunk carries its id; a passage its sequence range; a source its hot sections. */
export const isHit = (match: Match): match is Hit => 'chunk_id' in match
export const isPassage = (match: Match): match is Passage => 'seq_start' in match
export const isSource = (match: Match): match is Source => 'sections' in match

/** The heading a result sits under: a passage only knows its breadcrumb, whose last step it is. */
export function headingOf(match: Match): string {
  if (isPassage(match)) return match.header.split(' > ').at(-1) ?? ''
  return match.heading
}

/** Where the match sits in the document: page and chunk, page and chunk run, or the lines. */
export function position(match: Match): string {
  if (isHit(match) && match.page_start !== null) return `p. ${match.page_start} · chunk ${match.chunk_id}`
  if (isPassage(match)) {
    const run = match.seq_start === match.seq_end ? `chunk ${match.seq_start}` : `chunks ${match.seq_start}–${match.seq_end}`
    return match.page_start !== null ? `p. ${match.page_start} · ${run}` : `${run} · lines ${match.line_start}–${match.line_end}`
  }
  return `lines ${match.line_start}–${match.line_end}`
}

/** How full a result's bar is, against the others on screen: the best fills it, the weakest keeps
 *  a tenth, so a strong set and a weak one both read as a ranking. One result, or a tie, fills. */
export const MIN_FILL = 0.1
export function fillOf(score: number, best: number, worst: number): number {
  if (best === worst) return 1
  return MIN_FILL + (1 - MIN_FILL) * ((score - worst) / (best - worst))
}
