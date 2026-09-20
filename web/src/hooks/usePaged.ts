import { useCallback, useEffect, useRef, useState } from 'react'
import { DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, type Order, type Page, type PageRequest } from '../api'

export interface UsePagedOptions {
  pageSize?: number
  sort?: string
  order?: Order
  deps?: unknown[] // values that decide *what* is listed (a filter); a change starts over
}

export interface Paged<T> {
  items: T[]
  total: number | null
  hasMore: boolean
  loading: boolean
  error: string | null
  loadMore: () => void
  refresh: () => Promise<void>
  sort: string | undefined
  order: Order
  setSort: (field: string) => void
  pageSize: number
  setPageSize: (n: number) => void
}

// Cursor paging for a listing: one page at a time, `loadMore` appends the next one, `refresh`
// re-reads the rows already on screen in a single request (what a poll needs). A response is
// dropped once a newer request has started, so a slow first page cannot overwrite a newer sort.
export function usePaged<T>(fetchPage: (q: PageRequest) => Promise<Page<T>>, opts: UsePagedOptions = {}): Paged<T> {
  const [items, setItems] = useState<T[]>([])
  const [total, setTotal] = useState<number | null>(null)
  const [cursor, setCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(true) // the first page is requested on mount
  const [error, setError] = useState<string | null>(null)
  const [sort, setSortField] = useState<string | undefined>(opts.sort)
  const [order, setOrder] = useState<Order>(opts.order ?? 'asc')
  const [pageSize, setPageSizeState] = useState(opts.pageSize ?? DEFAULT_PAGE_SIZE)

  // callers pass an inline fetcher, so its identity changes every render: keep it out of the
  // dependencies and read the latest one when a request actually starts
  const fetchRef = useRef(fetchPage)
  const itemsRef = useRef<T[]>(items) // `refresh` sizes its window without re-creating itself
  const requestId = useRef(0)
  useEffect(() => {
    fetchRef.current = fetchPage
  })

  // Every update lands when the request settles: an effect starts the first load, and a
  // synchronous state update from an effect is a cascading render. `loading` is raised by the
  // event that caused the load instead.
  const load = useCallback((query: PageRequest, append: boolean): Promise<void> => {
    const id = ++requestId.current
    const newest = () => id === requestId.current
    return fetchRef.current(query).then(
      (page) => {
        if (!newest()) return
        itemsRef.current = append ? [...itemsRef.current, ...page.items] : page.items
        setItems(itemsRef.current)
        setTotal(page.total)
        setCursor(page.next_cursor)
        setError(null)
        setLoading(false)
      },
      (e: unknown) => {
        if (!newest()) return
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      },
    )
  }, [])

  const reset = useCallback(() => load({ cursor: null, page_size: pageSize, sort, order }, false), [load, pageSize, sort, order])

  // One request wide enough to cover everything already loaded. Polling uses it, so it leaves
  // `loading` alone and the rows in place until the new ones arrive.
  const refresh = useCallback(() => {
    const visible = Math.min(MAX_PAGE_SIZE, Math.max(pageSize, itemsRef.current.length))
    return load({ cursor: null, page_size: visible, sort, order }, false)
  }, [load, pageSize, sort, order])

  const loadMore = useCallback(() => {
    if (cursor === null || loading) return
    setLoading(true)
    void load({ cursor, page_size: pageSize, sort, order }, true)
  }, [load, cursor, loading, pageSize, sort, order])

  const setSort = useCallback(
    (field: string) => {
      setOrder(field === sort && order === 'asc' ? 'desc' : 'asc') // same field toggles, a new one starts ascending
      setSortField(field)
      setLoading(true)
    },
    [sort, order],
  )

  const setPageSize = useCallback((n: number) => {
    setPageSizeState(Math.min(MAX_PAGE_SIZE, Math.max(1, n)))
    setLoading(true)
  }, [])

  // the first page, and a fresh first page whenever sort, order, page size or the filters change
  const depsKey = JSON.stringify(opts.deps ?? [])
  useEffect(() => {
    void reset()
  }, [reset, depsKey])

  return {
    items,
    total,
    hasMore: cursor !== null,
    loading,
    error,
    loadMore,
    refresh,
    sort,
    order,
    setSort,
    pageSize,
    setPageSize,
  }
}
