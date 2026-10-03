import type { BulkKind, OperationProgress } from '../api'
import { useOptions } from '../hooks/useOptions'

// What a whole-thing operation is doing while it runs, in the words of the button that started it.
const DOING: Record<BulkKind, string> = {
  index_collection: 'queueing documents',
  delete_collection: 'deleting',
  delete_document: 'deleting',
  summarize_document: 'describing',
  summarize_collection: 'describing',
  create_backup: 'backing up',
  restore_backup: 'restoring',
}

/** How an operation started from a button stands: running, or how it ended and why. */
export function BulkStatus({ operation }: { operation: OperationProgress }) {
  const running = useOptions().active_run_statuses.includes(operation.status)
  const what = DOING[operation.kind]
  return (
    <span className="muted">
      {running ? `${what}…` : `${what}: ${operation.status.toLowerCase()}`}
      {operation.progress !== null && ` ${operation.progress.done}/${operation.progress.total}`}
      {operation.error !== null && ` — ${operation.error}`}
    </span>
  )
}
