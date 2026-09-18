import type { Paged } from '../hooks/usePaged'

const PAGE_SIZES = [50, 100, 500]

// Footer of a paged list: how much is loaded, the next page, and how big a page is.
export function Pager({ paged }: { paged: Paged<unknown> }) {
  const { total, hasMore, loading, pageSize } = paged
  const loaded = paged.items.length
  return (
    <div className="pager">
      <span className="muted">
        {loaded} of {total ?? '?'}
      </span>
      {hasMore && (
        <>
          <span className="muted">·</span>
          <button onClick={paged.loadMore} disabled={loading}>
            {loading ? 'loading…' : 'Load more'}
          </button>
        </>
      )}
      <span className="muted">·</span>
      <label className="muted">
        page size{' '}
        <select value={pageSize} onChange={(e) => paged.setPageSize(Number(e.target.value))}>
          {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
          {!PAGE_SIZES.includes(pageSize) && <option value={pageSize}>{pageSize}</option>}
        </select>
      </label>
    </div>
  )
}

// Clickable column header; the arrow shows the direction of the sort that is in effect.
export function SortHeader({ field, label, paged }: { field: string; label: string; paged: Paged<unknown> }) {
  const { sort, order } = paged
  const active = sort === field
  return (
    <th className={active ? 'sortable active' : 'sortable'}>
      <button onClick={() => paged.setSort(field)} title={`sort by ${label}`}>
        {label}
        {active && <span className="sort-arrow">{order === 'asc' ? '▲' : '▼'}</span>}
      </button>
    </th>
  )
}
