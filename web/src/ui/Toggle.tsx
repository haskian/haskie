interface BoxProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  id?: string // some design pages style a specific box, e.g. #bg-classic
}

/** A switch: on or off, applied at once. */
export function Toggle({ label, checked, onChange, disabled, id }: BoxProps) {
  return (
    <label className="toggle">
      <input
        type="checkbox"
        role="switch"
        id={id}
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      {label}
    </label>
  )
}

/** A checkbox: one item of a set. */
export function Check({ label, checked, onChange, disabled, id }: BoxProps) {
  return (
    <label className="check">
      <input type="checkbox" id={id} checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  )
}
