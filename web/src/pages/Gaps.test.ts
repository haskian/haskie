import { describe, expect, test } from 'bun:test'
import type { GapQuestion, GapTopic, LoggedResult, ReplayedGap } from '../api'
import { bands, nearText, questionSub, replayText, signalText, topicHint, topicId, topicSub } from './gaps/group'

const NOW = 1_790_000_000
const near: LoggedResult = {
  position: 0,
  parent: null,
  relation: null,
  collection: 'distributed-systems',
  document: 'ddia.pdf',
  seq_start: 3,
  seq_end: 4,
  line_start: 10233,
  line_end: 10240,
  header: 'Part III > Stream Processing',
  location: 'ddia.pdf p.482 L10233-10240',
  score: 0.031,
}
const question: GapQuestion = {
  id: 41,
  search_id: 17,
  ts: NOW - 7200,
  session_id: 'claude-code a3f9',
  actor: 'mcp',
  tool: 'excerpts',
  question: 'How should a background job retry a failed Kafka message without duplicates?',
  context: 'a Python service on Kafka',
  collections: ['distributed-systems', 'kafka'],
  signal: 'weak',
  result_count: 25,
  best_similarity: 0.712,
  best_rerank: null,
  near_misses: [near],
}
const topic: GapTopic = {
  question: question.question,
  questions: [question, { ...question, id: 40, ts: NOW - 10800, session_id: 'cursor 7b21', signal: 'uncovered', near_misses: [] }],
  sessions: 2,
  first_at: NOW - 10800,
  last_at: NOW - 7200,
  collections: ['distributed-systems', 'kafka'],
}
const once: GapTopic = { ...topic, question: 'tone mapping', questions: [{ ...question, id: 7, signal: 'empty' }], sessions: 0, collections: [] }

describe('bands', () => {
  const cases: Array<{ name: string; value: GapTopic[]; expected: Array<[string, string[]]> }> = [
    { name: 'nothing', value: [], expected: [] },
    { name: 'asked once only', value: [once], expected: [['Asked once', ['7']]] },
    { name: 'both bands, in the order given', value: [once, topic], expected: [['Asked again', ['41']], ['Asked once', ['7']]] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(bands(one.value).map((group) => [group.label, group.items.map(topicId)])).toEqual(one.expected)
    })
  }
})

describe('topic lines', () => {
  test('several questions and sessions', () => {
    expect(topicSub(topic, NOW)).toBe('2 times · 2 sessions · 2 h ago')
    expect(topicHint(topic)).toBe('weak match · no excerpt answers it · distributed-systems · kafka')
  })
  test('one question, no session, no collection', () => {
    expect(topicSub({ ...once, sessions: 1 }, NOW)).toBe('1 time · 1 session · 2 h ago')
    expect(topicSub(once, NOW)).toBe('1 time · no session · 2 h ago')
    expect(topicHint(once)).toBe('nothing came back · no collections')
  })
})

describe('signalText', () => {
  const cases: Array<{ name: string; value: Parameters<typeof signalText>[0]; expected: string }> = [
    { name: 'nothing came back', value: { signal: 'empty', best_rerank: null, best_similarity: null }, expected: 'nothing came back' },
    { name: 'asked with others, answered by none', value: { signal: 'uncovered', best_rerank: 0.9, best_similarity: 0.9 }, expected: 'no excerpt answers it' },
    { name: 'the reranker decided', value: { signal: 'weak', best_rerank: 0.0312, best_similarity: 0.9 }, expected: 'weak match: reranker 0.03' },
    { name: 'the cosine decided', value: { signal: 'weak', best_rerank: null, best_similarity: 0.712 }, expected: 'weak match: cosine 0.71' },
    { name: 'no score kept', value: { signal: 'weak', best_rerank: null, best_similarity: null }, expected: 'weak match' },
    { name: 'answered', value: { signal: null, best_rerank: 0.9, best_similarity: 0.9 }, expected: 'answered' },
  ]
  for (const one of cases) test(one.name, () => expect(signalText(one.value)).toBe(one.expected))
})

describe('questionSub', () => {
  const cases: Array<{ name: string; value: GapQuestion; expected: string }> = [
    { name: 'a tool with a session', value: question, expected: 'weak match: cosine 0.71 · via excerpts · claude-code a3f9 · 2 h ago' },
    { name: 'no session', value: { ...question, tool: 'sources', session_id: null }, expected: 'weak match: cosine 0.71 · via sources · no session · 2 h ago' },
  ]
  for (const one of cases) test(one.name, () => expect(questionSub(one.value, NOW)).toBe(one.expected))
})

describe('nearText', () => {
  test('nothing came back', () => expect(nearText([])).toBeNull())
  test('the closest by citation', () =>
    expect(nearText([near, { ...near, location: 'kafka-notes.md L12-30' }])).toBe('closest: ddia.pdf p.482 L10233-10240 · kafka-notes.md L12-30'))
})

describe('replayText', () => {
  const again: ReplayedGap = { id: 41, question: question.question, signal: null, result_count: 3, best_similarity: 0.83, best_rerank: null, results: [near] }
  const cases: Array<{ name: string; value: ReplayedGap; expected: string }> = [
    { name: 'answered now', value: again, expected: 'now answered: ddia.pdf p.482 L10233-10240' },
    {
      name: 'answered, every place cited in order',
      value: { ...again, results: [near, { ...near, location: 'kafka.md L5-7' }] },
      expected: 'now answered: ddia.pdf p.482 L10233-10240 · kafka.md L5-7',
    },
    { name: 'answered with nothing to cite', value: { ...again, results: [] }, expected: 'now answered' },
    { name: 'still nothing', value: { ...again, signal: 'empty', result_count: 0, results: [] }, expected: 'now: still nothing came back' },
    { name: 'still weak', value: { ...again, signal: 'weak', best_similarity: 0.7 }, expected: 'now: still weak match: cosine 0.70' },
  ]
  for (const one of cases) test(one.name, () => expect(replayText(one.value)).toBe(one.expected))
})
