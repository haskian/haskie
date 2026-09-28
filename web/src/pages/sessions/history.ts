import type { SessionEvent } from '../../api'

/** The line under a history row's subject: what the action came to, in the words of its kind. */
export function historySub(event: SessionEvent): string {
  const { detail } = event
  switch (event.action) {
    case 'search': {
      // the tool that ran it (`session.EventDetail.scope`)
      const where = detail.scope ? ` via ${detail.scope}` : ''
      const hits = detail.hits ?? 0
      if (detail.error) return `failed${where}: ${detail.error} · ${event.duration_ms} ms`
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
