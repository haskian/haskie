import type { Heading } from '../api'

/** Where a search result sits in its document: the heading it is under, and how far in it starts. */
export interface Anchor {
  heading: string
  offset?: number // char offset of the match in the markdown; absent for a source
}

/** The headings a table-of-contents entry sits under, outermost first, ending with the entry. */
export function headingPath(toc: Heading[], index: number): string[] {
  const trail: string[] = []
  let level = Infinity
  for (let at = index; at >= 0; at -= 1) {
    if (toc[at].level < level) {
      trail.unshift(toc[at].text)
      level = toc[at].level
    }
  }
  return trail
}

/**
 * The table-of-contents entry a result belongs to, as the index its rendered heading is anchored
 * by (`h-<index>`). Headings with the result's name are preferred; among those (or among all,
 * when none carries the name) the last one at or before the result wins.
 */
export function anchorIndex(toc: Heading[], anchor: Anchor): number | null {
  const indexed = toc.map((heading, index) => ({ heading, index }))
  const named = indexed.filter(({ heading }) => heading.text === anchor.heading)
  const pool = named.length > 0 ? named : indexed
  if (pool.length === 0) return null
  if (anchor.offset === undefined) return pool[0].index
  // The toc holds byte offsets and the match a char offset. Bytes never trail chars,
  // so "byte offset <= char offset" can only pick a heading too early, never too late; the name
  // filter above makes that the same heading in practice.
  const before = pool.filter(({ heading }) => heading.offset <= anchor.offset!)
  return (before.at(-1) ?? pool[0]).index
}
