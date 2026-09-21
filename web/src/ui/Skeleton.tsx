/** The shape of a document while it loads: a title, a byline, a page, a footer line. */
export function Skeleton() {
  return (
    <div className="skeleton" aria-hidden="true">
      <span className="skeleton-bar" />
      <span className="skeleton-dot" />
      <span className="skeleton-line" />
      <span className="skeleton-line short" />
      <span className="skeleton-block" />
      <span className="skeleton-bar" />
    </div>
  )
}
