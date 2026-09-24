import type { EmbeddingEntry } from '../../api'

/** One cached embedding as a line: the model, then every chunk setting its cache id is keyed by,
 *  so two entries of one document never read the same. The parser and the OCR choice are keyed
 *  too, but they belong to the document, so every entry of it shares them. */
export function embeddingLabel(entry: EmbeddingEntry): string {
  return [
    entry.model,
    `${entry.chunker} ${entry.chunk_size}`,
    `merge ${entry.chunk_merge_below}%`,
    entry.chunk_frame ? 'frame' : 'no frame',
    `${entry.rows} rows`,
  ].join(' · ')
}
