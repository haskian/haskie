import type { Operation, OperationKindSummary } from '../../api'
import { day } from '../../format'

export type GroupBy = 'status' | 'kind' | 'day'

// Run statuses, in the words the page groups them under. Anything else keeps its own name.
const STATUS_LABELS: Record<string, string> = {
  ENQUEUED: 'Running',
  PENDING: 'Running',
  SUCCESS: 'Completed',
  ERROR: 'Failed',
  CANCELLED: 'Cancelled',
}
const STATUS_ORDER: readonly string[] = ['Running', 'Completed', 'Failed', 'Cancelled']

export const statusGroup = (operation: Operation): string => STATUS_LABELS[operation.status] ?? operation.status

/** The day an operation started, without the time: "Sat 19 Jan". */
export const dayGroup = (operation: Operation): string => day(operation.created_at)

export interface OperationGroup {
  key: string
  operations: Operation[]
}

/**
 * The sections of the listing. Operations arrive newest first, which is the order within every
 * section and the order of the day sections; status and kind have an order of their own.
 */
export function groupOperations(operations: Operation[], by: GroupBy, kinds: OperationKindSummary[]): OperationGroup[] {
  if (by === 'kind') {
    return kinds
      .map((summary) => ({ key: summary.label, operations: operations.filter((one) => one.kind === summary.kind) }))
      .filter((group) => group.operations.length > 0)
  }
  const keyOf = by === 'status' ? statusGroup : dayGroup
  const groups = new Map<string, Operation[]>()
  for (const operation of operations) {
    const key = keyOf(operation)
    const bucket = groups.get(key)
    if (bucket) bucket.push(operation)
    else groups.set(key, [operation])
  }
  const sections = [...groups].map(([key, rows]) => ({ key, operations: rows }))
  if (by === 'day') return sections
  const rank = (key: string): number => (STATUS_ORDER.includes(key) ? STATUS_ORDER.indexOf(key) : STATUS_ORDER.length)
  return sections.sort((a, b) => rank(a.key) - rank(b.key))
}
