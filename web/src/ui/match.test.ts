import { describe, expect, test } from 'bun:test'
import { fillOf, MIN_FILL } from './match'

describe('fillOf', () => {
  const cases: Array<{ name: string; score: number; scores: number[]; expected: number }> = [
    { name: 'the best result fills the bar', score: 0.9, scores: [0.9, 0.5, 0.2], expected: 1 },
    { name: 'the weakest keeps the minimum', score: 0.2, scores: [0.9, 0.5, 0.2], expected: MIN_FILL },
    { name: 'the rest sit between, by score', score: 0.55, scores: [0.9, 0.55, 0.2], expected: 0.55 },
    { name: 'one result fills', score: 0.3, scores: [0.3], expected: 1 },
    { name: 'a tie fills every bar', score: 0.3, scores: [0.3, 0.3], expected: 1 },
    { name: 'raw BM25 scores above one still rank', score: 4, scores: [12, 4], expected: MIN_FILL },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(fillOf(one.score, Math.max(...one.scores), Math.min(...one.scores))).toBeCloseTo(one.expected, 10)
    })
  }
})
