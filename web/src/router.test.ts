import { describe, expect, test } from 'bun:test'
import { formatRoute, parseRoute, type Route } from './router'

const EXPLORE: Route = { name: 'explore' }

describe('parseRoute', () => {
  const cases: Array<{ name: string; hash: string; expected: Route }> = [
    { name: 'empty hash falls back to explore', hash: '', expected: EXPLORE },
    { name: 'bare hash falls back to explore', hash: '#', expected: EXPLORE },
    { name: 'root slash falls back to explore', hash: '#/', expected: EXPLORE },
    { name: 'unknown page falls back to explore', hash: '#/nowhere', expected: EXPLORE },
    { name: 'explore', hash: '#/explore', expected: EXPLORE },
    { name: 'documents', hash: '#/documents', expected: { name: 'documents', doc: undefined } },
    { name: 'documents with a name', hash: '#/documents/lamp.pdf', expected: { name: 'documents', doc: 'lamp.pdf' } },
    {
      name: 'documents with an encoded slash in the name',
      hash: '#/documents/renders%2Flamp.pdf',
      expected: { name: 'documents', doc: 'renders/lamp.pdf' },
    },
    {
      name: 'documents with an encoded space in the name',
      hash: '#/documents/area%20lights.md',
      expected: { name: 'documents', doc: 'area lights.md' },
    },
    { name: 'documents with a trailing slash has no name', hash: '#/documents/', expected: { name: 'documents', doc: undefined } },
    { name: 'collections', hash: '#/collections', expected: { name: 'collections', collection: undefined } },
    { name: 'collections with a name', hash: '#/collections/A%E2%80%93E', expected: { name: 'collections', collection: 'A–E' } },
    { name: 'operations', hash: '#/operations', expected: { name: 'operations' } },
    { name: 'sessions', hash: '#/sessions', expected: { name: 'sessions', session: undefined } },
    { name: 'sessions with an id', hash: '#/sessions/agent-1', expected: { name: 'sessions', session: 'agent-1' } },
    { name: 'insights', hash: '#/insights', expected: { name: 'insights' } },
    { name: 'settings', hash: '#/settings', expected: { name: 'settings' } },
    { name: 'a hash without the leading slash still parses', hash: '#settings', expected: { name: 'settings' } },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(parseRoute(testCase.hash)).toEqual(testCase.expected)
    })
  }
})

describe('formatRoute', () => {
  const cases: Array<{ name: string; route: Route; expected: string }> = [
    { name: 'explore', route: EXPLORE, expected: '#/explore' },
    { name: 'documents', route: { name: 'documents' }, expected: '#/documents' },
    { name: 'documents with a name', route: { name: 'documents', doc: 'lamp.pdf' }, expected: '#/documents/lamp.pdf' },
    {
      name: 'documents encodes a slash in the name',
      route: { name: 'documents', doc: 'renders/lamp.pdf' },
      expected: '#/documents/renders%2Flamp.pdf',
    },
    { name: 'collections', route: { name: 'collections' }, expected: '#/collections' },
    { name: 'collections encodes the name', route: { name: 'collections', collection: 'A–E' }, expected: '#/collections/A%E2%80%93E' },
    { name: 'operations', route: { name: 'operations' }, expected: '#/operations' },
    { name: 'sessions', route: { name: 'sessions' }, expected: '#/sessions' },
    { name: 'insights', route: { name: 'insights' }, expected: '#/insights' },
    { name: 'sessions with an id', route: { name: 'sessions', session: 'agent 1' }, expected: '#/sessions/agent%201' },
    { name: 'settings', route: { name: 'settings' }, expected: '#/settings' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(formatRoute(testCase.route)).toBe(testCase.expected)
    })
  }
})

describe('round trip', () => {
  const routes: Route[] = [
    EXPLORE,
    { name: 'documents' },
    { name: 'documents', doc: 'renders/lamp.pdf' },
    { name: 'collections' },
    { name: 'collections', collection: 'A–E' },
    { name: 'operations' },
    { name: 'sessions' },
    { name: 'sessions', session: 'agent 1' },
    { name: 'settings' },
  ]
  for (const route of routes) {
    test(formatRoute(route), () => {
      expect(parseRoute(formatRoute(route))).toEqual(route)
    })
  }
})
