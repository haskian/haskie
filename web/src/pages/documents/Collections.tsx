import { Library, Minus, Plus } from 'lucide-react'
import { href } from '../../router'
import { ListPane, ModalStatus } from '../../ui'

/** The collections that hold a document beside the ones that could, each with its own button:
 *  the collection modal's two panes, seen from the document. */
export function CollectionsTab({
  held,
  all,
  attachable,
  busy,
  onAttach,
  onDetach,
}: {
  held: string[]
  all: string[]
  // A collection takes a document only once it is imported; the backend refuses it before.
  attachable: boolean
  busy: boolean
  onAttach: (collection: string) => void
  onDetach: (collection: string) => void
}) {
  const available = all.filter((name) => !held.includes(name))
  return (
    <>
      {!attachable && <ModalStatus>A collection takes the document once it is imported.</ModalStatus>}
      <div className="split">
        <ListPane head="In collections" count={held.length}>
          {held.map((name) => (
            <li className="list-item" key={name}>
              <Library className="icon" />
              <span className="list-text">
                <a href={href({ name: 'collections', collection: name })}>{name}</a>
              </span>
              <button className="btn btn-ghost" type="button" aria-label="Remove" disabled={busy} onClick={() => onDetach(name)}>
                <Minus className="icon" />
              </button>
            </li>
          ))}
        </ListPane>
        <ListPane head="Available" count={available.length}>
          {available.map((name) => (
            <li className="list-item" key={name}>
              <Library className="icon" />
              <span className="list-text">{name}</span>
              <button className="btn btn-ghost" type="button" aria-label="Add" disabled={busy || !attachable} onClick={() => onAttach(name)}>
                <Plus className="icon" />
              </button>
            </li>
          ))}
        </ListPane>
      </div>
    </>
  )
}
