import type { Document, Member } from '../../api'

/**
 * Imported documents this collection does not hold yet.
 *
 * The ceiling is this client-side diff of two capped listings. Once a collection (or the home)
 * outgrows MAX_PAGE_SIZE, the server has to answer it with a `not_in` filter instead.
 */
export function candidateDocuments(imported: Document[], members: Member[]): Document[] {
  const held = new Set(members.map((member) => member.document.name))
  return imported.filter((doc) => !held.has(doc.name))
}

/** The collections other than `collection` that hold each imported document, by its name. */
export function otherCollections(imported: Document[], collection: string): Map<string, string[]> {
  return new Map(imported.map((doc) => [doc.name, doc.collections.filter((one) => one !== collection)]))
}
