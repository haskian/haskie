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

/** The documents one Import created, by name. */
export const importedNames = (results: PromiseSettledResult<ImportedDocument>[]): string[] =>
  results.flatMap((result) => (result.status === 'fulfilled' ? [result.value.name] : []))

/** What one Import leaves of the staged set: `sent` went out, `results` came back in its order.
 *  Applied by id to the set as it is now, so a file staged or renamed meanwhile stays as it is.
 *  Refused files stay with why, to be renamed or removed. */
export function waitingAfter(
  current: StagedFile[],
  sent: StagedFile[],
  results: PromiseSettledResult<ImportedDocument>[],
): StagedFile[] {
  const outcome = new Map(sent.map((one, at) => [one.staging_id, results[at]]))
  return current.flatMap((one) => {
    const result = outcome.get(one.staging_id)
    if (result === undefined) return [one]
    return result.status === 'fulfilled' ? [] : [{ ...one, error: errorText(result.reason) }]
  })
}

/** The files one Import sends: the same bytes are the same document, so a file already imported
 *  is left out. */
export const fresh = (files: StagedFile[]): StagedFile[] => files.filter((one) => one.duplicate === null)

/** The one button's label: the count once there is more than one file. */
export const importLabel = (count: number): string => (count === 1 ? 'Import' : `Import ${count} documents`)
