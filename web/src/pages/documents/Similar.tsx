import { Fragment, useCallback, useEffect, useState } from 'react'
import { api, type Document, type Similar } from '../../api'
import { errorText } from '../../format'
import { useOptions } from '../../hooks/useOptions'
import { usePoll } from '../../hooks/usePoll'
import { href } from '../../router'
import { documentIcon } from '../../ui'

const link = (name: string) => <a href={href({ name: 'documents', document: name })}>{name}</a>
const suffixOf = (name: string) => name.slice(name.lastIndexOf('.'))

/** The documents holding the very same file, each a link, and what that means for the reader. */
export function Duplicates({ names, advice = '' }: { names: string[]; advice?: string }) {
  return (
    <p className="notice">
      The same file is already imported as{' '}
      {names.map((name, at) => (
        <Fragment key={name}>
          {at > 0 && ', '}
          {link(name)}
        </Fragment>
      ))}
      .{advice && ` ${advice}`}
    </p>
  )
}

/** What a document may repeat: the same file under other names, then the nearest by content.
 *  Each name opens that document. */
export function SimilarDocuments({ similar }: { similar: Similar }) {
  return (
    <>
      {similar.identical.length > 0 && <Duplicates names={similar.identical} />}
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
