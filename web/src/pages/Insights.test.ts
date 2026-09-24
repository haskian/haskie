import { describe, expect, test } from 'bun:test'
import { MAX_SERIES, stacked, trend, type Point } from './insights/trend'

const OTHER = 'other sessions'

// A search is a point of weight one under its session; an index is a point of `chunks` under its
// collection. Timestamps are built from local dates so the day buckets line up whatever zone the
// test runs in.
const at = (day: number, hour: number): number => new Date(2026, 8, day, hour, 0, 0).getTime() / 1000
const NOW = at(21, 15)
const point = (ts: number, key = 'claude-code a3f9', n = 1): Point => ({ ts, key, n })

describe('trend', () => {
  const cases: Array<{ name: string; points: Point[]; days: number; expected: { buckets: number; series: Array<[string, number[]]>; total: number } }> = [
    { name: 'no searches: the buckets are still there', points: [], days: 7, expected: { buckets: 7, series: [], total: 0 } },
    {
      name: 'one search lands in its local day, the last bucket being today',
      points: [point(at(21, 9))],
      days: 7,
      expected: { buckets: 7, series: [['claude-code a3f9', [0, 0, 0, 0, 0, 0, 1]]], total: 1 },
    },
    {
      name: 'a search before the window is left out',
      points: [point(at(14, 9)), point(at(15, 0))],
      days: 7,
      expected: { buckets: 7, series: [['claude-code a3f9', [1, 0, 0, 0, 0, 0, 0]]], total: 1 },
    },
    {
      name: 'the busiest session leads; equal totals sort by id',
      points: [point(at(20, 9), 'b'), point(at(20, 10), 'b'), point(at(21, 9), 'a'), point(at(21, 9), 'c'), point(at(21, 9), 'c')],
      days: 7,
      expected: { buckets: 7, series: [['b', [0, 0, 0, 0, 0, 2, 0]], ['c', [0, 0, 0, 0, 0, 0, 2]], ['a', [0, 0, 0, 0, 0, 0, 1]]], total: 5 },
    },
    {
      name: 'one day is 24 hourly buckets ending in the current hour',
      points: [point(at(21, 15)), point(at(21, 14)), point(at(20, 16)), point(at(20, 15))],
      days: 1,
      expected: { buckets: 24, series: [['claude-code a3f9', [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]], total: 3 },
    },
    {
      name: 'a weighted point adds its whole weight: chunks by collection',
      points: [point(at(21, 9), 'A–E', 40), point(at(21, 12), 'A–E', 9), point(at(20, 9), 'K–O', 12)],
      days: 7,
      expected: { buckets: 7, series: [['A–E', [0, 0, 0, 0, 0, 0, 49]], ['K–O', [0, 0, 0, 0, 0, 12, 0]]], total: 61 },
    },
    {
      name: 'past six sessions, the sixth and later fold into one',
      points: ['s1', 's2', 's3', 's4', 's5', 's6', 's7'].flatMap((id, i) => Array.from({ length: 8 - i }, () => point(at(21, 9), id))),
      days: 1,
      expected: {
        buckets: 24,
        series: [['s1', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0]], ['s2', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0]], ['s3', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0]], ['s4', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]], ['s5', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0]], [OTHER, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]]],
        total: 35,
      },
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      const found = trend(one.points, one.days, NOW, OTHER)
      expect(found.buckets.length).toBe(one.expected.buckets)
      expect(found.buckets.at(-1)).toBe(one.days === 1 ? at(21, 15) : at(21, 0))
      expect(found.series.map((s) => [s.id, s.counts])).toEqual(one.expected.series)
      expect(found.series.length).toBeLessThanOrEqual(MAX_SERIES)
      expect(found.total).toBe(one.expected.total)
    })
  }
})

test('stacked columns run from the whole stack down to the last series', () => {
  const found = trend([point(at(21, 9), 'a'), point(at(21, 9), 'b'), point(at(21, 10), 'b')], 2, NOW, OTHER)
  expect(stacked(found)).toEqual([found.buckets, [0, 3], [0, 1]])
})
