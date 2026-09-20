import type { FieldDoc } from '../api'

// Label + definition for one setting; `doc` comes from /api/options.docs (single source of truth).
export function Field({ doc, children }: { doc: FieldDoc; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>
        <span className="field-title">{doc.title}</span>
        {children}
      </label>
      {doc.description && <p className="field-help muted">{doc.description}</p>}
    </div>
  )
}

// A numeric setting. `null` is what an emptied input reports: a form of overrides stores it as
// "use the default", one of concrete values reads it as 0.
export function NumberField({
  doc,
  value,
  min,
  step = 1,
  placeholder,
  onChange,
}: {
  doc: FieldDoc
  value: number | null
  min: number
  step?: number
  placeholder?: string
  onChange: (value: number | null) => void
}) {
  return (
    <Field doc={doc}>
      <input
        type="number"
        min={min}
        step={step}
        value={value ?? ''}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
    </Field>
  )
}
