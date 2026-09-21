import type { CollectionSummary } from '../../api'
import { groupByRange, type RangeGroup } from '../../ui'

/** One band per name range that holds a collection; empty bands are left out. */
export const groupByName = (collections: CollectionSummary[]): RangeGroup<CollectionSummary>[] =>
  groupByRange(collections, (one) => one.name)

/** The line under a tile's name: how much the collection holds, and how much of it is moving. */
export function tileSub(collection: CollectionSummary): string {
  const documents = `${collection.counts.total} documents`
  return collection.counts.active > 0 ? `${documents} · ${collection.counts.active} active` : documents
}
