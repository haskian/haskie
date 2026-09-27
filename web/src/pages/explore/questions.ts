/** The questions an excerpts search asks: the one in the search box, then the other parts, one a
 *  line, blank lines and repeats left out, as the backend would leave them out. */
export function questionsOf(first: string, parts: string): string[] {
  const all = [first, ...parts.split('\n')].map((line) => line.trim()).filter(Boolean)
  return [...new Set(all)]
}
