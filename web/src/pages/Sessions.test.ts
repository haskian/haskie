import { describe, expect, test } from 'bun:test'
import type { SessionEvent, SessionSummary } from '../api'
import { groupByStatus, tileSub } from './sessions/group'
import { historySub, searchLines, type SearchLine } from './sessions/history'

const search: SessionEvent = {
  ts: 1_758_460_000,
  action: 'search',
  subject: 'shadow mapping',
  detail: { scope: 'explore', hits: 12, documents: ['lamp.pdf', 'cube.pdf'], questions: ['shadow mapping'] },
  operation_id: null,
  duration_ms: 48,
}

describe('historySub', () => {
  const cases: Array<{ name: string; value: SessionEvent; expected: string }> = [
    { name: 'search with hits', value: search, expected: '12 hits in 2 documents · 48 ms' },
    { name: 'search with no hits', value: { ...search, detail: { scope: 'text', hits: 0, documents: [] } }, expected: 'no hits · 48 ms' },
    {
      name: 'a failed search says why',
      value: { ...search, detail: { scope: 'excerpts', hits: 0, documents: [], error: 'NotFound: collection not found: ghost' } },
      expected: 'failed: NotFound: collection not found: ghost · 48 ms',
    },
    { name: 'import names its operation', value: { ...search, action: 'import', subject: 'lamp.pdf', detail: {}, operation_id: 'imp:lamp.pdf:1' }, expected: 'imported, conversion queued' },
    { name: 'attach names the collection', value: { ...search, action: 'attach', subject: 'lamp.pdf', detail: { collection: 'A–E' }, operation_id: 'idx:1' }, expected: 'attached to A–E, index queued' },
    { name: 'detach names the collection', value: { ...search, action: 'detach', subject: 'lamp.pdf', detail: { collection: 'A–E' } }, expected: 'removed from A–E' },
    { name: 'describe', value: { ...search, action: 'describe', subject: 'lamp.pdf', detail: {} }, expected: 'description set' },
    { name: 'collections chosen', value: { ...search, action: 'collections', subject: 'A–E, K–O', detail: { collections: ['A–E', 'K–O'] } }, expected: 'searches these collections' },
    { name: 'collections cleared', value: { ...search, action: 'collections', subject: '', detail: { collections: [] } }, expected: 'searches no collections' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(historySub(one.value)).toBe(one.expected)
    })
  }
})

const asked: SessionEvent = {
  ...search,
  subject: 'How do shadow maps alias? | Why does PCF soften them?',
  detail: {
    scope: 'excerpts',
    hits: 4,
    documents: ['lamp.pdf'],
    questions: ['How do shadow maps alias?', 'Why does PCF soften them?'],
    context: 'A real-time renderer for a lamp scene',
  },
}

describe('searchLines', () => {
  const cases: Array<{ name: string; value: SessionEvent; expected: SearchLine[] }> = [
    {
      name: 'context first, then each question numbered',
      value: asked,
      expected: [
        { tag: 'Context', text: 'A real-time renderer for a lamp scene', context: true },
        { tag: 'Q1', text: 'How do shadow maps alias?', context: false },
        { tag: 'Q2', text: 'Why does PCF soften them?', context: false },
      ],
    },
    { name: 'no context, one question', value: search, expected: [{ tag: 'Q1', text: 'shadow mapping', context: false }] },
    { name: 'an empty context gets no line', value: { ...asked, detail: { ...asked.detail, context: '' } }, expected: searchLines(asked).slice(1) },
    { name: 'no questions recorded', value: { ...asked, detail: { scope: 'excerpts', hits: 0, documents: [] } }, expected: [] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(searchLines(one.value)).toEqual(one.expected)
    })
  }
})

const NOW = 1_758_460_000
const fresh: SessionSummary = { id: 'claude-code a3f9', collections: ['A–E', 'K–O'], last_at: NOW - 120 }
const stale: SessionSummary = { id: 'cursor 7b21', collections: ['P–T'], last_at: NOW - 3600 }
const older: SessionSummary = { id: 'agent 91dd', collections: [], last_at: NOW - 7200 }
const unused: SessionSummary = { id: 'claude-desktop c04e', collections: [], last_at: null }

describe('groupByStatus', () => {
  const cases: Array<{ name: string; value: SessionSummary[]; expected: Array<[string, string[]]> }> = [
    { name: 'nothing', value: [], expected: [] },
    { name: 'seen within the window is active', value: [fresh], expected: [['Active', ['claude-code a3f9']]] },
    { name: 'at the edge of the window is still active', value: [{ ...fresh, last_at: NOW - 900 }], expected: [['Active', ['claude-code a3f9']]] },
    { name: 'past the window is idle', value: [{ ...fresh, last_at: NOW - 901 }], expected: [['Idle', ['claude-code a3f9']]] },
    { name: 'never used is idle and last', value: [unused, older], expected: [['Idle', ['agent 91dd', 'claude-desktop c04e']]] },
    { name: 'both bands, newest first in each', value: [unused, older, fresh, stale], expected: [['Active', ['claude-code a3f9']], ['Idle', ['cursor 7b21', 'agent 91dd', 'claude-desktop c04e']]] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(groupByStatus(one.value, NOW).map((group) => [group.label, group.items.map((item) => item.id)])).toEqual(one.expected)
    })
  }
})

describe('tileSub', () => {
  const cases: Array<{ name: string; value: SessionSummary; expected: string }> = [
    { name: 'never used', value: unused, expected: 'no collections · never used' },
    { name: 'seen with collections', value: fresh, expected: '2 collections · 2 min ago' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(tileSub(one.value, NOW)).toBe(one.expected)
    })
  }
})
