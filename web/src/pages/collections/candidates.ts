import type { Document, Member } from '../../api'

/**
 * Imported documents this collection does not hold yet.
 *
 * ponytail: the ceiling is this client-side diff of two capped listings — once a collection (or
 * the home) outgrows MAX_PAGE_SIZE, the server has to answer it with a `not_in` filter instead.
 */
export function candidateDocuments(imported: Document[], members: Member[]): Document[] {
  const held = new Set(members.map((member) => member.document.name))
  return imported.filter((doc) => !held.has(doc.name))
}
