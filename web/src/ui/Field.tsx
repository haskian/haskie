import type { ReactNode } from 'react'

/**
 * A labelled control. The label is a sibling, not a wrapper, because the controls it holds
 * (`Picker`, `Toggle`, `Check`) already carry a label of their own.
 */
export function Field({ label, help, children }: { label: string; help?: string; children: ReactNode }) {
  return (
    <div className="field">
      <span className="label">{label}</span>
      {children}
      {help !== undefined && help !== '' && <span className="faint">{help}</span>}
    </div>
  )
}
