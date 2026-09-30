import { useState, type FormEvent } from 'react'

/** A name, editable, with a Rename button. Mount it keyed by the name, so a rename starts the next
 *  draft afresh. A form, so Enter renames the way the browser already does it. */
export function RenameForm({ name, label, busy, onRename }: { name: string; label: string; busy: boolean; onRename: (to: string) => void }) {
  const [draft, setDraft] = useState(name)
  const to = draft.trim()
  const submit = (event: FormEvent) => {
    event.preventDefault()
    onRename(to)
  }
  return (
    <form className="input-group" onSubmit={submit}>
      <input className="input" aria-label={label} value={draft} onChange={(event) => setDraft(event.target.value)} />
      <button className="btn" type="submit" disabled={busy || to === '' || to === name}>
        Rename
      </button>
    </form>
  )
}
