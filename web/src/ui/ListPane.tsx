import type { ReactNode } from 'react'

/** One pane of a two-pane membership editor: a head with its count, and the rows as a list. */
export function ListPane({ head, count, children }: { head: string; count: number; children: ReactNode }) {
  return (
    <section className="pane">
      <span className="pane-head mono muted">{`${head} · ${count}`}</span>
      <div className="pane-body">
        <ul className="list">{children}</ul>
      </div>
    </section>
  )
}
