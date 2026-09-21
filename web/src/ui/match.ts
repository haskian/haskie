import type { DocumentMatch, Hit } from '../api'

/** Either shape a search result arrives in: a passage that matched, or a document that did. */
export type Match = Hit | DocumentMatch

/** Only a passage carries a chunk; a document match knows the lines and nothing finer. */
export const isHit = (match: Match): match is Hit => 'chunk_id' in match

/** Where the match sits in the document: the page and chunk when they are known, else the lines. */
export function position(match: Match): string {
  if (isHit(match) && match.page_start !== null) return `p. ${match.page_start} · chunk ${match.chunk_id}`
  return `lines ${match.line_start}–${match.line_end}`
}

/** How full a result's bar is, against the others on screen: the best fills it, the weakest keeps
 *  a tenth, so a strong set and a weak one both read as a ranking. One result, or a tie, fills. */
export const MIN_FILL = 0.1
export function fillOf(score: number, best: number, worst: number): number {
  if (best === worst) return 1
  return MIN_FILL + (1 - MIN_FILL) * ((score - worst) / (best - worst))
}
