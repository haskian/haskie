import { Info as InfoIcon } from 'lucide-react'
import type { ReactNode } from 'react'

/** A line that says how things are: an empty list, what produced a result, a hint. In a box
 *  tinted gray with an info icon first. An error is not this: it stays a plain line. */
export function Info({ children }: { children: ReactNode }) {
  return (
    <p className="info">
      <InfoIcon className="icon" />
      <span>{children}</span>
    </p>
  )
}
