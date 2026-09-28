import { useEffect, useRef, useState } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { api } from '../api'
import type { PageProps } from '../App'
import { errorText } from '../format'
import { Shell, Tabs, type TabDef } from '../ui'
import { DAY, HOUR, stacked, trend, type Point, type Trend } from './insights/trend'

const PLOT_HEIGHT = 240
// Greys, darkest for the biggest series: a session or a document is an identity, so its shade holds across ranges.
const GREYS = ['#0f1317', '#333940', '#575e66', '#7b828b', '#9fa5ad', '#c0c5cb']

// The tab id is the range in days, so the selected tab and the loaded range are one number.
const RANGES: TabDef[] = [
  { id: '1', label: '1d' },
  { id: '7', label: '7d' },
  { id: '14', label: '14d' },
  { id: '30', label: '1m' },
  { id: '90', label: '3m' },
]
const DEFAULT_DAYS = 7

const css = (name: string): string => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

const bucketLabel = (unixSeconds: number, byHour: boolean): string =>
  new Date(unixSeconds * 1000).toLocaleString('en-GB', byHour ? { dateStyle: 'medium', timeStyle: 'short' } : { dateStyle: 'medium' })

// Module-level, so a chart's effect sees one loader across renders.
// a search made without a session is a series of its own
const loadSearches = (days: number): Promise<Point[]> =>
  api.searchTrend(days).then((points) => points.map((one) => ({ ts: one.ts, key: one.session_id ?? 'no session', n: 1 })))
const loadChunks = (days: number): Promise<Point[]> =>
  api
    .chunkTrend(days)
    // an import writes to no collection: it embeds the document for the ones that index it later
    .then((points) => points.map((one) => ({ ts: one.ts, key: one.document, n: one.chunks, detail: one.collection ?? 'import' })))

/** Two trends over the same ranges: searches stacked by session, and indexed chunks stacked by
 *  document, an import and every index of it alike: how much the agents use the shelf, and how
 *  much lands on it. */
export function Insights({ route, counts }: PageProps) {
  return (
    <Shell current={route.name} counts={counts}>
      <div className="insights sections">
        <TrendChart what="searches" other="other sessions" load={loadSearches} />
        <TrendChart what="chunks indexed" other="other documents" load={loadChunks} />
      </div>
    </Shell>
  )
}

/** One stacked bar chart with its own range tabs: a big number, the bars, and a legend. */
function TrendChart({ what, other, load }: { what: string; other: string; load: (days: number) => Promise<Point[]> }) {
  const [days, setDays] = useState(DEFAULT_DAYS)
  const [found, setFound] = useState<Trend | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hovered, setHovered] = useState<number | null>(null) // the bucket under the cursor

  useEffect(() => {
    let live = true
    load(days)
      .then((points) => {
        if (live) setFound(trend(points, days, Date.now() / 1000, other))
      })
      .catch((cause: unknown) => setError(errorText(cause)))
    return () => {
      live = false
    }
  }, [days, load, other])

  const at = hovered !== null && found !== null && hovered < found.buckets.length ? hovered : null

  return (
    <section className="chart">
      <div className="chart-head">
        <div className="stat stat-lg">
          <span className="value">{found === null ? '—' : at === null ? found.total : found.series.reduce((sum, one) => sum + one.counts[at], 0)}</span>
          <span className="label label-mono">{found !== null && at !== null ? bucketLabel(found.buckets[at], found.byHour) : `${what} · ${RANGES.find((one) => one.id === String(days))?.label}`}</span>
        </div>
        <Tabs tabs={RANGES} selected={String(days)} onSelect={(id) => setDays(Number(id))} />
      </div>
      {error !== null && <p className="muted">{error}</p>}
      {found !== null && <Plot found={found} onHover={setHovered} />}
      {found !== null && (
        <div className="legend">
          {found.series.map((one, j) => (
            <span key={one.id} className="legend-item" title={one.details.length > 0 ? one.details.join(', ') : undefined}>
              <i style={{ background: GREYS[j] }} />
              {one.id}
              <b>{at === null ? one.total : one.counts[at]}</b>
            </span>
          ))}
        </div>
      )}
    </section>
  )
}

// The chart is rebuilt whenever the data changes: a uPlot is cheap to make, and the series list
// (one per session) changes with the range, which `setData` alone cannot follow.
function Plot({ found, onHover }: { found: Trend; onHover: (index: number | null) => void }) {
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const element = host.current
    if (!element) return
    const bars = uPlot.paths.bars?.({ size: [0.6, 24], gap: 2, radius: (_u, seriesIndex) => (seriesIndex === 1 ? [0.25, 0] : [0, 0]) })
    const chart = new uPlot(
      {
        width: element.clientWidth,
        height: PLOT_HEIGHT,
        legend: { show: false },
        scales: {
          // half a step of padding either side, so the first and last bars are whole
          x: {
            range: (u, min, max) => {
              const half = u.data[0].length > 1 ? (u.data[0][1] - u.data[0][0]) / 2 : DAY / 2
              return [min - half, max + half]
            },
          },
          y: { range: (_u, _min, max) => [0, Math.max(1, max)] },
        },
        cursor: { y: false, points: { show: false } },
        axes: [
          {
            stroke: css('--text-muted'),
            grid: { show: false },
            ticks: { show: false },
            font: `11px ${css('--font-mono')}`,
            incrs: [HOUR, 3 * HOUR, 6 * HOUR, DAY, 7 * DAY, 14 * DAY, 30 * DAY],
            values: (_u, splits) =>
              splits.map((t) =>
                new Date(t * 1000).toLocaleString('en-GB', found.byHour ? { hour: '2-digit', minute: '2-digit' } : { day: 'numeric', month: 'short' }),
              ),
          },
          {
            stroke: css('--text-muted'),
            grid: { stroke: css('--line-soft'), width: 1 },
            ticks: { show: false },
            font: `11px ${css('--font-mono')}`,
            // wide enough for the longest label: a chunk count grows past four digits and a comma
            size: (_u, values) => Math.max(32, 12 + 7 * Math.max(0, ...(values ?? []).map((one) => one.length))),
            splits: (_u, _axis, _min, max) => [0, Math.ceil(max / 2), Math.ceil(max)],
          },
        ],
        series: [{}, ...found.series.map((one, j) => ({ label: one.id, stroke: GREYS[j], fill: GREYS[j], width: 0, paths: bars, points: { show: false } }))],
        hooks: { setCursor: [(u) => onHover(u.cursor.idx ?? null)] },
      },
      stacked(found) as uPlot.AlignedData,
      element,
    )
    // Width only: the chart sets its own height, which would re-fire the observer in a loop.
    let width = element.clientWidth
    const fit = new ResizeObserver(() => {
      if (element.clientWidth === width) return
      width = element.clientWidth
      chart.setSize({ width, height: PLOT_HEIGHT })
    })
    fit.observe(element)
    return () => {
      fit.disconnect()
      chart.destroy()
    }
  }, [found, onHover])

  return <div className="chart-plot" ref={host} />
}
