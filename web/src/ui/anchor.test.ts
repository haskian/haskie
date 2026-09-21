import { describe, expect, test } from 'bun:test'
import type { Heading } from '../api'
import { anchorIndex, breadcrumb, type Anchor } from './anchor'

const toc: Heading[] = [
  { level: 1, text: 'Implementing Domain-Driven Design', offset: 0 },
  { level: 2, text: 'This page intentionally left blank', offset: 120 },
  { level: 2, text: '286 Chapter 8 DOMAIN EVENTS', offset: 661_000 },
  { level: 2, text: 'This page intentionally left blank', offset: 700_000 },
  { level: 2, text: '286 Chapter 8 DOMAIN EVENTS', offset: 900_000 },
]

describe('breadcrumb', () => {
  const nested: Heading[] = [
    { level: 1, text: 'Book', offset: 0 },
    { level: 2, text: 'Part I', offset: 10 },
    { level: 3, text: 'Chapter 1', offset: 20 },
    { level: 3, text: 'Chapter 2', offset: 30 },
    { level: 2, text: 'Part II', offset: 40 },
    { level: 4, text: 'Deep', offset: 50 },
  ]
  const cases: Array<{ name: string; index: number; expected: string[] }> = [
    { name: 'the first heading is its own trail', index: 0, expected: ['Book'] },
    { name: 'a sibling is skipped, the parents kept', index: 3, expected: ['Book', 'Part I', 'Chapter 2'] },
    { name: 'a later part does not pass through the earlier one', index: 4, expected: ['Book', 'Part II'] },
    { name: 'a skipped level still climbs to its parents', index: 5, expected: ['Book', 'Part II', 'Deep'] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(breadcrumb(nested, one.index)).toEqual(one.expected)
    })
  }
})

describe('anchorIndex', () => {
  const cases: Array<{ name: string; toc: Heading[]; anchor: Anchor; expected: number | null }> = [
    { name: 'empty toc', toc: [], anchor: { heading: 'x', offset: 5 }, expected: null },
    { name: 'one heading with the name', toc, anchor: { heading: 'Implementing Domain-Driven Design', offset: 50 }, expected: 0 },
    { name: 'repeated name, last one at or before the match', toc, anchor: { heading: '286 Chapter 8 DOMAIN EVENTS', offset: 661_760 }, expected: 2 },
    { name: 'repeated name, match after the second', toc, anchor: { heading: '286 Chapter 8 DOMAIN EVENTS', offset: 950_000 }, expected: 4 },
    { name: 'name found but every offset is later: the first named', toc, anchor: { heading: '286 Chapter 8 DOMAIN EVENTS', offset: 10 }, expected: 2 },
    { name: 'no offset: the first named', toc, anchor: { heading: 'This page intentionally left blank' }, expected: 1 },
    { name: 'unknown name: the last heading before the match', toc, anchor: { heading: 'Chunk heading differs', offset: 800_000 }, expected: 3 },
    { name: 'unknown name and no offset: the first heading', toc, anchor: { heading: '' }, expected: 0 },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(anchorIndex(one.toc, one.anchor)).toBe(one.expected)
    })
  }
})
