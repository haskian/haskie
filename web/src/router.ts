import { useMemo, useSyncExternalStore } from 'react'

// Hash routes, no library. A page owns one hash segment; a second segment names the row whose
// modal is open, so a modal is a real address and closing it is a navigation back to the page.
export type Route =
  | { name: 'explore' }
  | { name: 'documents'; document?: string }
  | { name: 'collections'; collection?: string }
  | { name: 'operations' }
  | { name: 'sessions'; session?: string }
  | { name: 'insights' }
  | { name: 'settings' }

export type RouteName = Route['name']

export const DEFAULT_ROUTE: Route = { name: 'explore' }

// One hash, split into the page segment and the optional name after it. Names are encoded, so a
// document called "renders/lamp.pdf" arrives as one segment.
export function parseRoute(hash: string): Route {
  const [page, param] = hash.replace(/^#\/?/, '').split('/')
  const name = param === undefined || param === '' ? undefined : decodeURIComponent(param)
  switch (page) {
    case 'documents':
      return { name: 'documents', document: name }
    case 'collections':
      return { name: 'collections', collection: name }
    case 'operations':
      return { name: 'operations' }
    case 'sessions':
      return { name: 'sessions', session: name }
    case 'insights':
      return { name: 'insights' }
    case 'settings':
      return { name: 'settings' }
    default:
      return DEFAULT_ROUTE
  }
}

export function formatRoute(route: Route): string {
  switch (route.name) {
    case 'documents':
      return route.document === undefined ? '#/documents' : `#/documents/${encodeURIComponent(route.document)}`
    case 'collections':
      return route.collection === undefined ? '#/collections' : `#/collections/${encodeURIComponent(route.collection)}`
    case 'sessions':
      return route.session === undefined ? '#/sessions' : `#/sessions/${encodeURIComponent(route.session)}`
    default:
      return `#/${route.name}`
  }
}

/** The `#/…` string for an `<a href>`. */
export const href = formatRoute

export function navigate(route: Route): void {
  location.hash = formatRoute(route)
}

const subscribe = (onChange: () => void): (() => void) => {
  addEventListener('hashchange', onChange)
  return () => removeEventListener('hashchange', onChange)
}

// The snapshot is the hash itself: `useSyncExternalStore` compares snapshots by identity, and a
// freshly parsed object is never identical to the last one.
const readHash = (): string => location.hash

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, readHash, () => '')
  return useMemo(() => parseRoute(hash), [hash])
}
