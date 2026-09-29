import type { ImportedDocument, Staged } from '../../api'
import { errorText } from '../../format'

/** An upload waiting to be imported: the name the document will get (the filename until edited),
 *  and why the server refused it last time, if it did. */
export type StagedFile = Staged & { name: string; error: string | null }

/** Staging every picked file at once: the ones that landed, ready to name, and a line for each one
 *  that did not. A failure costs its own file only. */
export function staged(
  files: File[],
  results: PromiseSettledResult<Staged>[],
): { added: StagedFile[]; failures: string[] } {
  const added: StagedFile[] = []
  const failures: string[] = []
  results.forEach((result, at) => {
    if (result.status === 'fulfilled') added.push({ ...result.value, name: result.value.filename, error: null })
    else failures.push(`${files[at].name}: ${errorText(result.reason)}`)
  })
  return { added, failures }
}

/** What one Import leaves of the staged set: `sent` went out, `results` came back in its order.
 *  Applied by id to the set as it is now, so a file staged or renamed meanwhile stays as it is.
 *  Refused files stay with why, to be renamed or removed; the names are the documents created. */
export function settled(
  current: StagedFile[],
  sent: StagedFile[],
  results: PromiseSettledResult<ImportedDocument>[],
): { waiting: StagedFile[]; imported: string[] } {
  const refused = new Map<string, string>()
  const done = new Set<string>()
  const imported: string[] = []
  sent.forEach((one, at) => {
    const result = results[at]
    if (result.status === 'fulfilled') {
      done.add(one.staging_id)
      imported.push(result.value.name)
    } else refused.set(one.staging_id, errorText(result.reason))
  })
  const waiting = current
    .filter((one) => !done.has(one.staging_id))
    .map((one) => (refused.has(one.staging_id) ? { ...one, error: refused.get(one.staging_id) ?? null } : one))
  return { waiting, imported }
}

/** The one button's label: the count once there is more than one file. */
export const importLabel = (count: number): string => (count === 1 ? 'Import' : `Import ${count} documents`)
