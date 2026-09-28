import { createElement, type ReactNode } from 'react'

const escapeRegExp = (term: string): string => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** One pattern that matches any of the query's terms, case-insensitively, as a capturing group so
 *  `split` keeps the matches at every odd index; null for a query with no terms. */
export function termPattern(query: string): RegExp | null {
  const terms = query.split(/\s+/).filter(Boolean).map(escapeRegExp)
  return terms.length === 0 ? null : new RegExp(`(${terms.join('|')})`, 'gi')
}

/**
 * `text` split on the query's terms, with every match wrapped in `<mark>`. Whitespace separates
 * terms; the search backends match them independently, so the highlight does too.
 */
export function markTerms(text: string, query: string): ReactNode[] {
  const pattern = termPattern(query)
  if (pattern === null) return [text]
  const parts = text.split(pattern)
  return parts.map((part, index) => (index % 2 === 1 ? createElement('mark', { key: index }, part) : part))
}
