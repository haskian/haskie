export const HOUR = 3600
export const DAY = 24 * HOUR
/** The chart has six greys; a seventh series and beyond fold into one "other" series. */
export const MAX_SERIES = 6

/** One thing that happened: when, under which series, and how much of it (one search, n chunks). */
export interface Point {
  ts: number // unix seconds
  key: string
  n: number
}

export interface Series {
  id: string
  counts: number[] // one per bucket
  total: number
}

export interface Trend {
  byHour: boolean // the one-day range; every other range is bucketed by day
  buckets: number[] // unix seconds at which each bucket starts, oldest first
  series: Series[] // busiest first
  total: number
}

// A day range is bucketed by local calendar day, so a bar is "Tuesday" to the reader; the one-day
// range is bucketed by hour. Both end at the bucket `now` falls in.
const bucketStart = (unixSeconds: number, byHour: boolean): number => {
  const date = new Date(unixSeconds * 1000)
  if (byHour) date.setMinutes(0, 0, 0)
  else date.setHours(0, 0, 0, 0)
  return date.getTime() / 1000
}

const nextBucket = (start: number, byHour: boolean): number => {
  const date = new Date(start * 1000)
  if (byHour) date.setHours(date.getHours() + 1)
  else date.setDate(date.getDate() + 1)
  return date.getTime() / 1000
}

/** Totals per bucket per series over the last `days` days, biggest series first; the series
 *  past the sixth fold into one named `other`. */
export function trend(points: Point[], days: number, now: number, other: string): Trend {
  const byHour = days === 1
  const count = byHour ? 24 : days
  const buckets: number[] = []
  const indexOf = new Map<number, number>() // bucket start to its column
  let start = bucketStart(now - (count - 1) * (byHour ? HOUR : DAY), byHour)
  for (let i = 0; i < count; i++) {
    buckets.push(start)
    indexOf.set(start, i)
    start = nextBucket(start, byHour)
  }

  const perKey = new Map<string, number[]>()
  for (const point of points) {
    const index = indexOf.get(bucketStart(point.ts, byHour))
    if (index === undefined) continue // outside the window
    let counts = perKey.get(point.key)
    if (!counts) {
      counts = new Array<number>(count).fill(0)
      perKey.set(point.key, counts)
    }
    counts[index] += point.n
  }

  const ranked = [...perKey]
    .map(([id, counts]) => ({ id, counts, total: counts.reduce((sum, n) => sum + n, 0) }))
    .sort((a, b) => b.total - a.total || a.id.localeCompare(b.id))
  const series = ranked.slice(0, ranked.length > MAX_SERIES ? MAX_SERIES - 1 : MAX_SERIES)
  const rest = ranked.slice(series.length)
  if (rest.length > 0) {
    const counts = buckets.map((_, i) => rest.reduce((sum, one) => sum + one.counts[i], 0))
    series.push({ id: other, counts, total: counts.reduce((sum, n) => sum + n, 0) })
  }
  return { byHour, buckets, series, total: series.reduce((sum, one) => sum + one.total, 0) }
}

/** uPlot's columns for a stacked bar chart: series j is the running top of series j..end, so
 *  drawing them in order paints each lower stack over the one above. */
export function stacked(found: Trend): number[][] {
  const tops: number[][] = []
  let above = found.buckets.map(() => 0)
  for (const one of [...found.series].reverse()) {
    above = above.map((sum, i) => sum + one.counts[i])
    tops.unshift(above)
  }
  return [found.buckets, ...tops]
}
