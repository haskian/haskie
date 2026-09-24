import type { SessionEvent } from '../../api'

// The scopes a search records when it spans a whole selection (`session.EventDetail.scope`), so
// the row names the tool that ran it. Any other scope is the one collection a search looked in.
const TOOLS: ReadonlySet<string> = new Set(['explore', 'excerpts', 'sources', 'text'])

/** The line under a history row's subject: what the action came to, in the words of its kind. */
export function historySub(event: SessionEvent): string {
  const { detail } = event
  switch (event.action) {
    case 'search': {
      const scope = detail.scope
      const where = !scope ? '' : TOOLS.has(scope) ? ` via ${scope}` : ` in ${scope}`
      const hits = detail.hits ?? 0
      const found = hits === 0 ? 'no hits' : `${hits} hits in ${detail.documents?.length ?? 0} documents`
      return `${found}${where} · ${event.duration_ms} ms`
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
