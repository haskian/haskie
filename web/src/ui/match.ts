import type { Hit, Passage, Source } from '../api'

/** Any shape a search result arrives in: a chunk, a passage (or excerpt), or a document. */
export type Match = Hit | Passage | Source

/** A chunk carries its `seq`; a passage its sequence range; a source its hot sections. */
export const isHit = (match: Match): match is Hit => 'seq' in match
export const isSource = (match: Match): match is Source => 'sections' in match

/** Between two headings of a heading path, as the backend's `chunk.HEADING_SEP` joins them. */
export const HEADING_SEP = ' > '

/** The last heading of a joined heading path (`header`): the one the text sits directly under. */
export const lastHeading = (header: string): string => header.split(HEADING_SEP).at(-1) ?? ''

/** The heading a result sits under: a chunk's is the last of its path, and a passage or a source
 *  only knows its header, whose last heading it is. */
export function headingOf(match: Match): string {
  return isHit(match) ? (match.headings.at(-1) ?? '') : lastHeading(match.header)
}

/** What each of the chunker's cut rules means (see `indexing/chunking.md`), for the edges of a
 *  chunk on screen. Typed by the API's own enum, so a new rule fails the build until it is named. */
export const CUT_REASONS: Record<Hit['start_reason'], string> = {
  edge: 'start or end of the text',
  heading: 'a heading starts a new section',
  paragraph: 'a blank line between paragraphs',
  length_block: 'chunk full, cut between two blocks',
  length_sentence: 'chunk full, cut between two sentences',
  length_oversize: 'chunk full, cut inside a piece longer than a chunk',
}

/** How big one part of a chunk is, in the three units a reader counts in. */
export interface Size {
  chars: number
  words: number
  pieces: number
}

/** The markdown a piece of a chunk came from (`segment.PieceType`). */
export type PieceType = NonNullable<Hit['layout']>[number]['type']

/** One piece of a chunk's text: a sentence or a block kept whole, and where it starts in the
 *  text, in code points. */
export interface ChunkPiece {
  type: PieceType
  text: string
  position: number
}

/** Each piece type as a reader names it. Typed by the API's own enum, so a new type fails the
 *  build until it is named. */
export const PIECE_NAMES: Record<PieceType, string> = {
  heading: 'Heading',
  text: 'Text',
  list: 'List item',
  quote: 'Quote',
  table: 'Table',
  code: 'Code',
  html: 'HTML',
  rule: 'Rule',
  metadata: 'Metadata',
}

// Words by Unicode's rules rather than spaces, so Chinese and Japanese count too.
const WORDS = new Intl.Segmenter(undefined, { granularity: 'word' })
const wordsIn = (text: string): number => Array.from(WORDS.segment(text)).filter((part) => part.isWordLike).length

/** What the models read ahead of a chunk's text, as the backend's `chunk.frame` writes it: the
 *  heading path and a blank line. */
export const frameOf = (hit: Hit): string => (hit.frame.length > 0 ? `${hit.frame.join(HEADING_SEP)}\n\n` : '')

/** How big a chunk is as it was embedded: its frame (one line, counted as one piece), its own
 *  text, and the two together. Characters are code points, as Python counts them. */
export function chunkSizes(hit: Hit): { frame: Size; text: Size; total: Size } {
  const prefix = frameOf(hit)
  const frame = { chars: Array.from(prefix).length, words: wordsIn(prefix), pieces: prefix ? 1 : 0 }
  const text = { chars: Array.from(hit.text).length, words: wordsIn(hit.text), pieces: piecesOf(hit).length }
  const total = { chars: frame.chars + text.chars, words: frame.words + text.words, pieces: frame.pieces + text.pieces }
  return { frame, text, total }
}

/** A chunk's text as the embedding model read it, after its `frame` (the heading path), cut back
 *  into its typed pieces at the positions of its `layout`. Positions are code points, as Python
 *  counts them, so the text is sliced by code point too; a position past the text is ignored, and
 *  text ahead of the first position is one piece of plain text. */
export function piecesOf(hit: Hit): ChunkPiece[] {
  const chars = Array.from(hit.text)
  const layout = (hit.layout ?? []).filter((piece) => piece.position >= 0 && piece.position < chars.length)
  const starts = layout[0]?.position === 0 ? layout : [{ type: 'text' as const, position: 0 }, ...layout]
  const pieces: ChunkPiece[] = []
  starts.forEach(({ type, position }, i) => {
    const end = i + 1 < starts.length ? starts[i + 1].position : chars.length
    if (end > position) pieces.push({ type, text: chars.slice(position, end).join(''), position })
  })
  return pieces
}

/** What the hover hint of a piece says under its type: where it starts and how big it is. */
export const pieceMeta = (piece: ChunkPiece): string =>
  `position ${piece.position} · ${Array.from(piece.text).length} chars · ${wordsIn(piece.text)} words`

/** A hot section's citation without the document name it repeats: "p.3 L7-43" out of
 *  "doc.pdf p.3 L7-43". The block naming the section already names the document once. */
export function cite(location: string, doc: string): string {
  return location.startsWith(doc) ? location.slice(doc.length).trim() : location
}

/** The chunks a match covers, by `seq` (the 1-based position in its document): one chunk's, or a
 *  passage's first and last. */
const seqRange = (match: Hit | Passage): [number, number] => (isHit(match) ? [match.seq, match.seq] : [match.seq_start, match.seq_end])

/** The chunks a quoted match covers, as a bare number for the bottom corner of the quote: one
 *  chunk's, a passage's run, or nothing for a source (which shows no quote). */
export function seqLabel(match: Match): string {
  if (isSource(match)) return ''
  const [first, last] = seqRange(match)
  return first === last ? `${first}` : `${first}–${last}`
}

/** Where the match sits in the document: page and chunk run, or chunk run and lines; a source,
 *  which covers no one run, only its lines. */
export function position(match: Match): string {
  if (isSource(match)) return `lines ${match.line_start}–${match.line_end}`
  const [first, last] = seqRange(match)
  const run = first === last ? `chunk ${first}` : `chunks ${first}–${last}`
  return match.page_start !== null ? `p. ${match.page_start} · ${run}` : `${run} · lines ${match.line_start}–${match.line_end}`
}

/** How full a result's bar is, against the others on screen: the best fills it, the weakest keeps
 *  a tenth, so a strong set and a weak one both read as a ranking. One result, or a tie, fills. */
export const MIN_FILL = 0.1
export function fillOf(score: number, best: number, worst: number): number {
  if (best === worst) return 1
  return MIN_FILL + (1 - MIN_FILL) * ((score - worst) / (best - worst))
}
