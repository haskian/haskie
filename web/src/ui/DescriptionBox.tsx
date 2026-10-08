import { useEffect, useRef } from 'react'

/**
 * A description that saves itself: on blur, and on unmount for the text a closing dialog (Escape
 * fires no blur) would otherwise drop. Uncontrolled, so a refresh of `value` after a save does not
 * fight the reader's typing. Saves only what changed.
 */
export function DescriptionBox({ value, placeholder, onSave }: { value: string; placeholder: string; onSave: (next: string) => void }) {
  const box = useRef<HTMLTextAreaElement>(null)
  const latest = useRef({ value, onSave })
  useEffect(() => {
    latest.current = { value, onSave }
  })
  // reads refs only, so the first render's closure serves the unmount as well
  const save = () => {
    const next = box.current?.value
    if (next !== undefined && next !== latest.current.value) latest.current.onSave(next)
  }
  useEffect(() => save, [])
  return <textarea ref={box} className="textarea description-box" aria-label="Description" defaultValue={value} placeholder={placeholder} onBlur={save} />
}
