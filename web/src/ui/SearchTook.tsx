/** One line above the results: what came back, and how long it took. Blank but present before
 *  the first query, so the results do not jump down when it arrives. */
export function SearchTook({ counts, ms }: { counts: string; ms: number | null }) {
  return <p className="mono muted">{ms === null ? '\u00a0' : `${counts} · ${ms} ms`}</p>
}
