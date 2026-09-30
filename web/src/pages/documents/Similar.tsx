import { useCallback, useEffect, useState } from 'react'
import { api, type Document, type Similar } from '../../api'
import { errorText } from '../../format'
import { useOptions } from '../../hooks/useOptions'
import { usePoll } from '../../hooks/usePoll'
import { href } from '../../router'
import { documentIcon } from '../../ui'

const link = (name: string) => <a href={href({ name: 'documents', document: name })}>{name}</a>
const suffixOf = (name: string) => name.slice(name.lastIndexOf('.'))

/** The document a staged file already is, as a link: the file is not imported again. */
export function Duplicate({ name }: { name: string }) {
  return (
    <p className="notice">
      The same file is already imported as {link(name)}. It is not imported again.
    </p>
  )
}

/** What a document may repeat: the documents nearest by content, each name opening it. The
 *  same file is never imported twice, so there are no identical ones to list. */
export function SimilarDocuments({ similar }: { similar: Similar }) {
  return (
    <>
      {similar.nearest.length === 0 ? (
        <p className="muted">Nothing to compare with: no other document has a vector under the current embedding model.</p>
      ) : (
        <ul className="list">
          {similar.nearest.map((one) => {
            const Icon = documentIcon(suffixOf(one.document))
            return (
              <li className="list-item" key={one.document}>
                <Icon className="icon" />
                <span className="list-text">
                  {link(one.document)}
                  <span className="sub">similarity {one.similarity.toFixed(2)}</span>
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </>
  )
}

/** The book just imported: its status while the pipeline runs, then what it may repeat. The
 *  nearest documents need its vector, so they come once the import is done. */
export function JustImported({ name }: { name: string }) {
  const options = useOptions()
  const [row, setRow] = useState<Document | null>(null)
  const [similar, setSimilar] = useState<Similar | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    api.document(name).then(setRow).catch((cause: unknown) => setError(errorText(cause)))
  }, [name])
  useEffect(refresh, [refresh])
  usePoll(row === null || options.active_document_statuses.includes(row.status), refresh)

  const imported = row?.status === 'imported'
  useEffect(() => {
    if (imported) api.similarDocuments(name).then(setSimilar).catch((cause: unknown) => setError(errorText(cause)))
  }, [imported, name])

  return (
    <div className="field">
      <span className="label">
        Imported · {link(name)}
        {row !== null && !imported && ` · ${row.status}`}
      </span>
      {row?.error != null && <p className="muted">{row.error}</p>}
      {!imported && row?.error == null && <p className="muted">The nearest documents show once it is converted and embedded.</p>}
      {similar !== null && <SimilarDocuments similar={similar} />}
      {error !== null && <p className="muted">{error}</p>}
    </div>
  )
}
