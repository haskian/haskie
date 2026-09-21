import { createElement, type ReactNode } from 'react'

const escapeRegExp = (term: string): string => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/**
 * `text` split on the query's terms, with every match wrapped in `<mark>`. Whitespace separates
 * terms; the search backends match them independently, so the highlight does too.
 */
export function markTerms(text: string, query: string): ReactNode[] {
  const terms = query.split(/\s+/).filter(Boolean).map(escapeRegExp)
  if (terms.length === 0) return [text]
  // A capturing group makes `split` keep the matches, at every odd index of the result.
  const parts = text.split(new RegExp(`(${terms.join('|')})`, 'gi'))
  return parts.map((part, index) => (index % 2 === 1 ? createElement('mark', { key: index }, part) : part))
}
