import { useCallback, useState } from 'react'
import { ACTIVE_JOB_STATUSES, api, type BulkJob, type BulkStarted } from '../api'
import { usePoll } from './usePoll'

export interface BulkJobFollower {
  job: BulkJob | null // the last answer the job gave, finished ones included
  running: boolean
  start: (fn: () => Promise<BulkStarted>) => Promise<void>
}

// Work the backend only accepts (202) and runs in the background. `start` sends the request and
// follows the job it returns until it ends, then hands the finished job to `onDone`. A failed
// request rejects, so the caller decides what a failed start means; a failed poll has no caller
// to answer to and goes to `onError`.
//
// The follower depends on the job id, not on the job object every tick replaces, so `usePoll`
// keeps one interval for the life of the job. Give `onDone` and `onError` a stable identity
// (`useCallback`), or that interval is torn down and rebuilt on every tick.
export function useBulkJob(onDone: (job: BulkJob) => void, onError: (message: string) => void): BulkJobFollower {
  const [job, setJob] = useState<BulkJob | null>(null)
  const running = job !== null && ACTIVE_JOB_STATUSES.has(job.status)
  const id = running ? job.id : null

  // one answer from the job, whoever asked for it: keep it on the page, and report the last one
  const settle = useCallback(
    (next: BulkJob) => {
      setJob(next)
      if (!ACTIVE_JOB_STATUSES.has(next.status)) onDone(next)
    },
    [onDone],
  )

  const follow = useCallback(() => {
    if (id === null) return
    api
      .jobProgress(id)
      .then(settle)
      .catch((e: unknown) => onError(String(e)))
  }, [id, settle, onError])
  usePoll(id !== null, follow)

  const start = useCallback(
    async (fn: () => Promise<BulkStarted>) => {
      const { job_id } = await fn()
      settle(await api.jobProgress(job_id))
    },
    [settle],
  )

  return { job, running, start }
}
