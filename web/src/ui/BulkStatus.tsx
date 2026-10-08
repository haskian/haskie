import type { BulkKind, OperationProgress } from '../api'
import { useOptions } from '../hooks/useOptions'
import { ModalStatus } from './ModalStatus'

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

/** How an operation ended and why. Its button shows progress while it runs. */
export function BulkStatus({ operation }: { operation: OperationProgress }) {
  const running = useOptions().active_run_statuses.includes(operation.status)
  if (running) return null
  const what = DOING[operation.kind]
  return (
    <ModalStatus tone={operation.status === 'SUCCESS' ? 'success' : operation.status === 'ERROR' ? 'error' : 'warning'}>
      {`${what}: ${operation.status.toLowerCase()}`}
      {operation.progress !== null && ` ${operation.progress.done}/${operation.progress.total}`}
      {operation.error !== null && ` — ${operation.error}`}
    </ModalStatus>
  )
}
