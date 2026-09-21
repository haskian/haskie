import { api, type Hit } from '../../api'

/** The picker's value for "every collection", and the prefix that marks a session. */
export const ALL_SCOPE = '*'
export const SESSION_PREFIX = 'session:'

/** What the scope picker's value stands for. */
export type Scope = { kind: 'all' } | { kind: 'collection'; name: string } | { kind: 'session'; id: string }

// ponytail: a collection literally named "session:x" would read as a session here. Two pickers, or
// a composite value, would rule that out; one prefix is enough for names people actually use.
export function parseScope(value: string): Scope {
  if (value === ALL_SCOPE) return { kind: 'all' }
  if (value.startsWith(SESSION_PREFIX)) return { kind: 'session', id: value.slice(SESSION_PREFIX.length) }
  return { kind: 'collection', name: value }
}

/** Which collections a scope covers, for the literature search. `undefined` means every one. */
export function scopeCollections(scope: Scope, sessions: Record<string, string[]>): string[] | undefined {
  switch (scope.kind) {
    case 'all':
      return undefined
    case 'collection':
      return [scope.name]
    case 'session':
      return sessions[scope.id] ?? []
  }
}

/** The passage search each scope answers with: one endpoint per scope, all returning hits. */
export function searchSections(scope: Scope, query: string): Promise<Hit[]> {
  switch (scope.kind) {
    case 'all':
      return api.searchText(query).then((page) => page.items)
    case 'collection':
      return api.searchCollection(scope.name, query)
    case 'session':
      return api.search(scope.id, query)
  }
}
