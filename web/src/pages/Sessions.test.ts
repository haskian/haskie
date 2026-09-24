import { describe, expect, test } from 'bun:test'
import type { SessionEvent, SessionSummary } from '../api'
import { groupByStatus, tileSub } from './sessions/group'
import { historySub } from './sessions/history'

const search: SessionEvent = {
  ts: 1_758_460_000,
  action: 'search',
  subject: 'shadow mapping',
  detail: { scope: 'explore', hits: 12, documents: ['lamp.pdf', 'cube.pdf'] },
  operation_id: null,
  duration_ms: 48,
}

describe('historySub', () => {
  const cases: Array<{ name: string; value: SessionEvent; expected: string }> = [
    { name: 'exploration with hits', value: search, expected: '12 hits in 2 documents via explore · 48 ms' },
    { name: 'search with no hits', value: { ...search, detail: { scope: 'text', hits: 0, documents: [] } }, expected: 'no hits via text · 48 ms' },
    { name: 'excerpts name their tool', value: { ...search, detail: { scope: 'excerpts', hits: 2, documents: ['a', 'b'] } }, expected: '2 hits in 2 documents via excerpts · 48 ms' },
    { name: 'sources name their tool', value: { ...search, detail: { scope: 'sources', hits: 2, documents: ['a', 'b'] } }, expected: '2 hits in 2 documents via sources · 48 ms' },
    { name: 'a collection named like no tool is a collection', value: { ...search, detail: { scope: 'documents', hits: 2, documents: ['a', 'b'] } }, expected: '2 hits in 2 documents in documents · 48 ms' },
    { name: 'one collection', value: { ...search, detail: { scope: 'A–E', hits: 3, documents: ['a'] } }, expected: '3 hits in 1 documents in A–E · 48 ms' },
    { name: 'a search that recorded no scope', value: { ...search, detail: { hits: 1, documents: ['a'] } }, expected: '1 hits in 1 documents · 48 ms' },
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
