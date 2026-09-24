import { useCallback, useState } from 'react'
import { api, type BulkStarted, type OperationProgress } from '../api'
import { useOptions } from './useOptions'
import { usePoll } from './usePoll'

export interface OperationFollower {
  operation: OperationProgress | null // the last answer it gave, finished ones included
  running: boolean
  start: (fn: () => Promise<BulkStarted>) => Promise<void>
}

// Work the backend only accepts (202) and runs in the background. `start` sends the request and
// follows the operation it returns until it ends, then hands the finished one to `onDone`. A
// failed request rejects, so the caller decides what a failed start means; a failed poll has no
// caller to answer to and goes to `onError`.
//
// The follower depends on the operation id, not on the object every tick replaces, so `usePoll`
// keeps one interval for the life of the operation. Give `onDone` and `onError` a stable identity
// (`useCallback`), or that interval is torn down and rebuilt on every tick.
export function useOperation(onDone: (operation: OperationProgress) => void, onError: (message: string) => void): OperationFollower {
  const { active_run_statuses } = useOptions()
  const [operation, setOperation] = useState<OperationProgress | null>(null)
  const running = operation !== null && active_run_statuses.includes(operation.status)
  const id = running ? operation.id : null

  // one answer from the operation, whoever asked for it: keep it on the page, and report the last
  const settle = useCallback(
    (next: OperationProgress) => {
      setOperation(next)
      if (!active_run_statuses.includes(next.status)) onDone(next)
    },
    [active_run_statuses, onDone],
  )

  const follow = useCallback(() => {
    if (id === null) return
    api
      .operationProgress(id)
      .then(settle)
      .catch((e: unknown) => onError(String(e)))
  }, [id, settle, onError])
  usePoll(id !== null, follow)

  const start = useCallback(
    async (fn: () => Promise<BulkStarted>) => {
      const { operation_id } = await fn()
      settle(await api.operationProgress(operation_id))
    },
    [settle],
  )

  return { operation, running, start }
}
