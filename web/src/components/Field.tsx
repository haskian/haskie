import type { FieldDoc } from '../api'

// Label + definition for one setting; `doc` comes from /api/options.docs (single source of truth).
export function Field({ doc, name, children }: { doc?: FieldDoc; name: string; children: React.ReactNode }) {
  return (
    <div className="field">
      <label>
        <span className="field-title">{doc?.title ?? name}</span>
        {children}
      </label>
      {doc?.description && <p className="field-help muted">{doc.description}</p>}
    </div>
  )
}
