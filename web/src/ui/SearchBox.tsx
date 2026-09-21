import { LoaderCircle, X } from 'lucide-react'
import type { ReactNode } from 'react'

/**
 * The one search box, everywhere a page or a panel searches or filters: a scope picker in front
 * when there is one, the input, and a clear behind it that turns into a loader while a query runs.
 * A filter passes `onChange` alone and narrows as the reader types; a search also passes
 * `onSubmit`, which Enter runs the way the browser already does it.
 */
export function SearchBox({
  value,
  onChange,
  placeholder,
  onSubmit,
  onClear,
  busy = false,
  scope,
  id,
}: {
  value: string
  onChange: (value: string) => void
  placeholder: string
  onSubmit?: () => void
  onClear?: () => void // what clearing means beyond emptying the box: dropping results, say
  busy?: boolean
  scope?: ReactNode
  id?: string
}) {
  const clear = onClear ?? (() => onChange(''))
  return (
    <form
      className="input-group"
      role="search"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit?.()
      }}
    >
      {scope}
      <input className="input" id={id} type="search" placeholder={placeholder} value={value} onChange={(event) => onChange(event.target.value)} />
      {busy ? (
        <span className="btn btn-ghost" role="status" aria-label="Searching">
          <LoaderCircle className="icon spin spin-fast" />
        </span>
      ) : (
        <button className="btn btn-ghost" type="button" aria-label="Clear" onClick={clear}>
          <X className="icon" />
        </button>
      )}
    </form>
  )
}
