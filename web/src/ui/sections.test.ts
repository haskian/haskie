import { describe, expect, test } from 'bun:test'
import { keywordsOf, outlineIndex, type Spot } from './sections'

describe('keywordsOf', () => {
  const cases: Array<{ name: string; keywords: string[]; distinct: string[]; expected: Array<[string, boolean]> }> = [
    { name: 'the distinct first, marked, then the rest', keywords: ['saga', 'compensation', 'step'], distinct: ['step'], expected: [['step', true], ['saga', false], ['compensation', false]] },
    { name: 'no distinct: every keyword plain', keywords: ['saga'], distinct: [], expected: [['saga', false]] },
    { name: 'no outline: nothing to show', keywords: [], distinct: [], expected: [] },
  ]
  test.each(cases)('$name', ({ keywords, distinct, expected }) => {
    expect(keywordsOf({ keywords, distinct }).map(({ word, distinct: marked }): [string, boolean] => [word, marked])).toEqual(expected)
  })
})

describe('outlineIndex', () => {
  const OUTLINE: Spot[] = [
    { header: '', line_start: 1, line_end: 90 },
    { header: 'Sagas', line_start: 3, line_end: 40 },
    { header: 'Sagas > Retries', line_start: 20, line_end: 40 },
    { header: 'Quorums', line_start: 41, line_end: 60 },
    { header: 'Sagas', line_start: 61, line_end: 90 }, // the path again, after another section
  ]
  const cases: Array<{ name: string; section: Spot; expected: number; outline?: Spot[] }> = [
    { name: 'the same path and lines', section: { header: 'Sagas', line_start: 3, line_end: 40 }, expected: 1 },
    { name: 'a few lines apart: still that section', section: { header: 'Sagas > Retries', line_start: 22, line_end: 42 }, expected: 2 },
    { name: 'a path twice: the one its lines overlap most', section: { header: 'Sagas', line_start: 60, line_end: 88 }, expected: 4 },
    { name: 'the whole document', section: { header: '', line_start: 1, line_end: 90 }, expected: 0 },
    { name: 'a path the outline lacks', section: { header: 'Leaders', line_start: 3, line_end: 40 }, expected: -1 },
    { name: 'the same path, no shared line', section: { header: 'Quorums', line_start: 91, line_end: 99 }, expected: -1 },
    { name: 'no outline yet', section: { header: 'Sagas', line_start: 3, line_end: 40 }, expected: -1, outline: [] },
  ]
  test.each(cases)('$name', ({ section, expected, outline }) => {
    expect(outlineIndex(outline ?? OUTLINE, section)).toBe(expected)
  })
})
