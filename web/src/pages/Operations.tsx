import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MAX_PAGE_SIZE, api, type Operation as OperationRow, type OperationKind, type OperationKindSummary, type Task } from '../api'
import type { PageProps } from '../App'
import { useOptions } from '../hooks/useOptions'
import { usePoll } from '../hooks/usePoll'
import { Picker, Shell } from '../ui'
import './Operations.css'
import { groupOperations, type GroupBy } from './operations/group'
import { Operation } from './operations/Operation'
import { errorText } from '../format'

const PAGE_SIZE = 20 // one page per kind, as the backend lists them
// The kinds whose operations belong to one collection. The others ignore the filter (a model
// download is no collection's work), so they stay listed while it is set.
const BY_COLLECTION: ReadonlySet<OperationKind> = new Set<OperationKind>(['document', 'collection', 'maintenance'])

const GROUPS: { id: GroupBy; label: string }[] = [
  { id: 'status', label: 'Status' },
  { id: 'kind', label: 'Operation' },
  { id: 'day', label: 'Day' },
]

/** Every task of an operation: its jobs' tasks, one read per distinct job. */
const operationTasks = (operation: OperationRow): Promise<Task[]> =>
  Promise.all([...new Set(operation.jobs.map((job) => job.id))].map((id) => api.jobTasks(id))).then((lists) => lists.flat())

/**
 * Every kind of background work in one list: merged, sorted newest first, and grouped by status,
 * kind or day. While anything is running, every poll re-reads the whole listing.
 */
export function Operations({ route, counts }: PageProps) {
  const { active_run_statuses: active } = useOptions()
  const [kinds, setKinds] = useState<OperationKindSummary[]>([])
  const [operations, setOperations] = useState<OperationRow[]>([])
  const [tasks, setTasks] = useState<Record<string, Task[]>>({})
  const [hasMore, setHasMore] = useState(false)
  const [pages, setPages] = useState(1)
  const [collections, setCollections] = useState<string[]>([])
  const [collection, setCollection] = useState('')
  const [groupBy, setGroupBy] = useState<GroupBy>('status')
  const [error, setError] = useState<string | null>(null)
  const stale = useRef(new Set<string>()) // operations whose tasks were last read while they ran

  // A task listing that fails leaves the operation without one rather than failing the whole page.
  const loadTasks = useCallback(async (wanted: OperationRow[]): Promise<void> => {
    if (wanted.length === 0) return
    const loaded = await Promise.all(
      wanted.map((operation) =>
        operationTasks(operation)
          .then((rows) => [operation.id, rows] as const)
          .catch(() => [operation.id, []] as const),
      ),
    )
    for (const operation of wanted) {
      if (active.includes(operation.status)) stale.current.add(operation.id)
      else stale.current.delete(operation.id)
    }
    setTasks((current) => ({ ...current, ...Object.fromEntries(loaded) }))
  }, [active])

  // no cursor bookkeeping. The backend's cursor is an offset, so asking for a wider
  // first page is the same request as paging into it, and one window size serves every kind.
  const load = useCallback((): Promise<void> => {
    const pageSize = Math.min(MAX_PAGE_SIZE, PAGE_SIZE * pages)
    const filter = collection === '' ? undefined : collection
    return api
      .operationKinds()
      .then((summaries) =>
        Promise.all(
          summaries.map((summary) =>
            api.operations(summary.kind, { page_size: pageSize, collection: BY_COLLECTION.has(summary.kind) ? filter : undefined }),
          ),
        ).then((listings) => {
          const merged = listings.flatMap((listing) => listing.items).sort((a, b) => b.created_at - a.created_at)
          setKinds(summaries)
          setOperations(merged)
          setHasMore(listings.some((listing) => listing.next_cursor !== null))
          setError(null)
          return loadTasks(
            merged.filter((one) => one.kind === 'document' && (active.includes(one.status) || stale.current.has(one.id))),
          )
        }),
      )
      .catch((failure: unknown) => setError(errorText(failure)))
  }, [pages, collection, loadTasks, active])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    api.collectionNames().then(setCollections).catch(() => undefined)
  }, [])

  // The backend's count sees operations no page holds yet; the rows on screen keep the poll going
  // while one of them finishes.
  const running = kinds.reduce((n, summary) => n + summary.active, 0) + operations.filter((one) => active.includes(one.status)).length
  usePoll(running > 0, load)

  // An active operation's tasks are read by every poll. A finished one's are read by the click
  // that unfolds it, and once more by the poll that sees it finish, or the last in-flight list
  // would stay on screen.
  const toggle = useCallback((operation: OperationRow, open: boolean): void => {
    if (open && operation.kind === 'document') void loadTasks([operation])
  }, [loadTasks])

  const cancel = useCallback(
    (operation: OperationRow): void => {
      api
        .cancelOperation(operation.id)
        .then(load)
        .catch((failure: unknown) => setError(errorText(failure)))
    },
    [load],
  )

  const groups = useMemo(() => groupOperations(operations, groupBy, kinds), [operations, groupBy, kinds])

  const side = (
    <>
      <section>
        <span className="label label-mono">Group by</span>
        <nav className="nav" id="group-by">
          {GROUPS.map((group) => (
            <button
              key={group.id}
              className="nav-item"
              type="button"
              data-group={group.id}
              aria-current={group.id === groupBy ? 'true' : undefined}
              onClick={() => setGroupBy(group.id)}
            >
              {group.label}
            </button>
          ))}
        </nav>
      </section>
      <section>
        <span className="label label-mono">Collection</span>
        <Picker
          ariaLabel="Collection"
          options={[{ value: '', label: 'All collections' }, ...collections.map((name) => ({ value: name, label: name }))]}
          value={collection}
          onChange={setCollection}
        />
      </section>
    </>
  )

  return (
    <Shell current={route.name} counts={counts} side={side}>
      <div className="operations sections">
        {error !== null && <p className="muted">{error}</p>}
        {error === null && groups.length === 0 && <p className="muted">No operations yet</p>}
        {groups.map((group) => (
          <section className="operation-section section" key={group.key} data-status={group.key}>
            <span className="mono muted">{group.key}</span>
            {group.operations.map((operation) => (
              <Operation key={operation.id} operation={operation} tasks={tasks[operation.id] ?? null} onToggle={toggle} onCancel={cancel} />
            ))}
          </section>
        ))}
        {hasMore && (
          <div className="row">
            <button className="btn btn-ghost" type="button" onClick={() => setPages((page) => page + 1)}>
              Load more
            </button>
          </div>
        )}
      </div>
    </Shell>
  )
}
