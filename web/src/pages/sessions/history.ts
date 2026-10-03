import type { SessionEvent } from '../../api'

/** One tagged line of a search row: its context, then each question numbered as the results
 *  number them ("Q2"). */
export type SearchLine = { tag: string; text: string; context: boolean }

/** What a search asked, one line per part: the context it sent, if any, then its questions. */
export function searchLines(event: SessionEvent): SearchLine[] {
  const { context, questions } = event.detail
  return [
    ...(context ? [{ tag: 'Context', text: context, context: true }] : []),
    ...(questions ?? []).map((question, at) => ({ tag: `Q${at + 1}`, text: question, context: false })),
  ]
}

/** The line under a history row's subject: what the action came to, in the words of its kind. */
export function historySub(event: SessionEvent): string {
  const { detail } = event
  switch (event.action) {
    case 'search': {
      const hits = detail.hits ?? 0
      const took = `${event.duration_ms} ms`
      if (detail.error) return `failed: ${detail.error} · ${took}`
      const found = hits === 0 ? 'no hits' : `${hits} hits in ${detail.documents?.length ?? 0} documents`
      return `${found} · ${took}`
    }
    case 'import':
      return 'imported, conversion queued'
    case 'attach':
      return `attached to ${detail.collection}, index queued`
    case 'detach':
      return `removed from ${detail.collection}`
    case 'describe':
      return 'description set'
    case 'collections':
      return event.subject === '' ? 'searches no collections' : 'searches these collections'
  }
}
