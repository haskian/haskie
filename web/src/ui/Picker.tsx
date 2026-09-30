import { ChevronDown } from 'lucide-react'
import { useEffect, useRef } from 'react'

export interface PickerOption<T extends string> {
  value: T
  label: string
  sub?: string
  tag?: { text: string; title: string } // a short mark beside the label, and what it means on hover
}

/** An option's label, with its tag beside it when it has one. */
function Label<T extends string>({ option }: { option: PickerOption<T> }) {
  if (option.tag === undefined) return <span>{option.label}</span>
  return (
    <span className="picker-label">
      <span>{option.label}</span>
      <span className="picker-tag" title={option.tag.title}>
        {option.tag.text}
      </span>
    </span>
  )
}

/**
 * A `<details>` dropdown: the summary shows the pick, the list is the options. Controlled by
 * `value`; the open state stays with the element, the way the design page leaves it.
 */
export function Picker<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: PickerOption<T>[]
  value: T
  onChange: (value: T) => void
  ariaLabel?: string
}) {
  const details = useRef<HTMLDetailsElement>(null)
  const selected = options.find((option) => option.value === value)

  // A click anywhere else closes the list, as in `design.js`. The listener is on the document
  // because the click that closes the picker usually lands on another control entirely.
  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      const element = details.current
      if (element?.open && !element.contains(event.target as Node)) element.open = false
    }
    document.addEventListener('click', closeOnOutsideClick)
    return () => document.removeEventListener('click', closeOnOutsideClick)
  }, [])

  const pick = (next: T) => {
    onChange(next)
    if (details.current) details.current.open = false
  }

  return (
    <details className="picker" ref={details}>
      <summary aria-label={ariaLabel}>
        <span className="picker-value">
          {selected === undefined ? <span /> : <Label option={selected} />}
          {selected?.sub !== undefined && <span className="sub">{selected.sub}</span>}
        </span>
        <ChevronDown className="icon" />
      </summary>
      <ul className="picker-list" role="listbox" aria-label={ariaLabel}>
        {options.map((option) => (
          <li
            key={option.value}
            role="option"
            tabIndex={0}
            aria-selected={option.value === value}
            onClick={() => pick(option.value)}
            onKeyDown={(event) => {
              if (event.key !== 'Enter' && event.key !== ' ') return
              event.preventDefault()
              pick(option.value)
            }}
          >
            <Label option={option} />
            {option.sub !== undefined && <span className="sub">{option.sub}</span>}
          </li>
        ))}
      </ul>
    </details>
  )
}
