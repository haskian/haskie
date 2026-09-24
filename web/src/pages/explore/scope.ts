import type { SearchScope } from '../../api'

/** The picker's value for "every collection", and the prefix that marks a session. */
export const ALL_SCOPE = '*'
export const SESSION_PREFIX = 'session:'

/** What the scope picker's value stands for. */
export type Scope = { kind: 'all' } | { kind: 'collection'; name: string } | { kind: 'session'; id: string }

// a collection literally named "session:x" would read as a session here. Two pickers, or
// a composite value, would rule that out; one prefix is enough for names people actually use.
export function parseScope(value: string): Scope {
  if (value === ALL_SCOPE) return { kind: 'all' }
  if (value.startsWith(SESSION_PREFIX)) return { kind: 'session', id: value.slice(SESSION_PREFIX.length) }
  return { kind: 'collection', name: value }
}

/** The scope as the search endpoints take it. A session is sent by id: the backend resolves its
 *  selection, and one that selected nothing searches everything, as it would over MCP. */
export function scopeParams(scope: Scope): SearchScope {
  switch (scope.kind) {
    case 'all':
      return {}
    case 'collection':
      return { collections: [scope.name] }
    case 'session':
      return { session_id: scope.id }
  }
}
