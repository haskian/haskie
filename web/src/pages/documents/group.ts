import type { Document, DocumentStatus } from '../../api'
import { day } from '../../format'
import type { RangeGroup } from '../../ui'

const capitalise = (word: string): string => word.charAt(0).toUpperCase() + word.slice(1)

/** A document in no collection first, as no search reaches it; newest first after that. */
export const unfiledFirst = (a: Document, b: Document): number =>
  Number(a.collections.length > 0) - Number(b.collections.length > 0) || b.created_at - a.created_at

/** One band per status that has documents, in the order of `statuses`: the backend's
 *  `Options.document_statuses`, which is the import pipeline's own. */
export function groupByStatus(docs: Document[], statuses: readonly DocumentStatus[]): RangeGroup<Document>[] {
  return statuses.map((status) => ({
    label: capitalise(status),
    items: docs.filter((doc) => doc.status === status),
  })).filter((group) => group.items.length > 0)
}

/** One band per day something was imported, newest day first, and in it `unfiledFirst`. */
export function groupByDay(docs: Document[]): RangeGroup<Document>[] {
  const newestFirst = [...docs].sort((a, b) => b.created_at - a.created_at)
  const groups = new Map<string, Document[]>()
  for (const doc of newestFirst) {
    const label = day(doc.created_at)
    const bucket = groups.get(label)
    if (bucket) bucket.push(doc)
    else groups.set(label, [doc])
  }
  return [...groups].map(([label, items]) => ({ label, items: items.sort(unfiledFirst) }))
}

