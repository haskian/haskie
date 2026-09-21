import { Check, CircleDashed, X } from 'lucide-react'
import { memo, type CSSProperties } from 'react'
import { ACTIVE_JOB_STATUSES, type JobRow, type Task } from '../../api'
import { dateTime, duration } from '../../format'
import { Kv, Stages } from '../../ui'
import { bulkKind, count, endsOf, stageDefs, stageInfo, stagesFor, tagOf, taskState, taskText, type StageDef } from './stages'

const TASK_ICONS = { done: Check, error: X, todo: CircleDashed } as const

/** One job: the summary is head plus progress strip, the body is what it is made of. Memoised
 *  with stable callbacks, so a poll re-renders only the rows whose job or batches changed. */
export const Job = memo(function Job({
  job,
  tasks,
  onToggle,
  onCancel,
}: {
  job: JobRow
  tasks: Task[] | null
  onToggle: (job: JobRow, open: boolean) => void
  onCancel: (job: JobRow) => void
}) {
  const active = ACTIVE_JOB_STATUSES.has(job.status)
  const rows = tasks ?? []
  const byTask = job.kind === 'document' && rows.length > 0

  return (
    <details className="operation" onToggle={(event) => onToggle(job, event.currentTarget.open)}>
      <summary>
        <div className="operation-head">
          <span className="tag">
            <span className="kind">{tagOf(job)}</span>
            {/* the title is shortened when it does not fit, so the whole of it is the tooltip */}
            <span title={job.title}>{job.title}</span>
          </span>
          {/* the session whose action started it comes first: it is what the other columns lack */}
          <span className="mono muted">
            {job.origin !== null && `${job.origin} · `}
            {dateTime(job.created_at)} · {duration(job.updated_at - job.created_at)}
          </span>
        </div>
        <Stages stages={stagesFor(job, tasks)} variant={active ? 'glass' : 'line'} stripes={active} />
      </summary>
      <div className="operation-tasks">
        {byTask ? (
          stageDefs(job).map((def) => <TaskColumn key={def.stage} def={def} rows={rows.filter((task) => task.stage === def.stage)} />)
        ) : (
          <div className="operation-stage">
            <Kv rows={summaryRows(job)} />
          </div>
        )}
      </div>
      {(job.error !== null || active) && (
        <div className="operation-tasks">
          <div className="operation-stage">
            {job.error !== null && <pre className="md">{job.error}</pre>}
            {active && (
              <div className="row">
                <button className="btn btn-ghost" type="button" onClick={() => onCancel(job)}>
                  <X className="icon" />
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </details>
  )
})

/**
 * What a job that reports no batches has to say: its status and whatever its kind counts. The
 * skipped documents and the loaded model are on the strip above, so they are not repeated here.
 */
function summaryRows(job: JobRow): [string, string][] {
  const rows: [string, string][] = [['Status', job.status.toLowerCase()]]
  if (job.kind === 'document') rows.push(['Tasks', `${count(job, 'tasks_done')}/${count(job, 'tasks_total')}`])
  if (job.kind === 'collection' && bulkKind(job) === 'index_collection') rows.push(['Queued', `${count(job, 'done')} of ${count(job, 'total')}`])
  return rows
}

function TaskColumn({ def, rows }: { def: StageDef; rows: Task[] }) {
  const info = stageInfo(def.stage, rows)
  const { head, hidden, tail } = endsOf(rows)

  return (
    <div className="operation-stage" style={{ '--weight': def.weight } as CSSProperties}>
      <span className="label">{def.label}</span>
      <span className="operation-info">{info}</span>
      <ul className="tasks">
        {head.map(taskItem)}
        {hidden > 0 && <li className="task muted">+{hidden} more</li>}
        {tail.map(taskItem)}
      </ul>
    </div>
  )
}

function taskItem(task: Task) {
  const state = taskState(task)
  const Icon = TASK_ICONS[state]
  return (
    <li key={task.id} className={`task ${state}`} title={task.error ?? undefined}>
      <Icon className="icon" />
      {taskText(task)}
    </li>
  )
}

