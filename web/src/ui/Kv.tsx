import { Fragment, type ReactNode } from 'react'

/** A definition list of label/value pairs. Labels are unique within one list, so they key it. */
export function Kv({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([label, value]) => (
        <Fragment key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </Fragment>
      ))}
    </dl>
  )
}
