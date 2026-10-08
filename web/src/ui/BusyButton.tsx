import { LoaderCircle } from 'lucide-react'
import type { ButtonHTMLAttributes } from 'react'
import type { OperationProgress } from '../api'

/** Keep an action's progress inside its button, from the request through the background work. */
export function BusyButton({ busy, busyLabel, progress, disabled, children, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {
  busy: boolean
  busyLabel: string
  progress?: OperationProgress['progress']
}) {
  return (
    <button {...props} disabled={disabled || busy} aria-busy={busy || undefined}>
      {busy ? (
        <>
          <LoaderCircle className="icon spin spin-fast" aria-hidden="true" />
          {busyLabel}
          {progress != null && ` ${progress.done}/${progress.total}`}
        </>
      ) : children}
    </button>
  )
}
