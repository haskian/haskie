import { Check, CircleDashed, Download, X } from 'lucide-react'
import { memo, type CSSProperties } from 'react'
import { api, type Operation as OperationRow, type Task } from '../../api'
import { bytes, dateTime, duration } from '../../format'
import { useOptions } from '../../hooks/useOptions'
import { Jobs, Kv } from '../../ui'
import { bulkKind, count, endsOf, jobDefs, jobsFor, stageInfo, tagOf, taskState, taskText, type StageDef } from './jobs'

const TASK_ICONS = { done: Check, error: X, todo: CircleDashed } as const

/** One operation: the summary is head plus progress strip, the body is the jobs it is made of.
 *  Memoised with stable callbacks, so a poll re-renders only the rows whose operation or tasks
 *  changed. */
export const Operation = memo(function Operation({
  operation,
  tasks,
  onToggle,
  onCancel,
}: {
  operation: OperationRow
  tasks: Task[] | null
  onToggle: (operation: OperationRow, open: boolean) => void
  onCancel: (operation: OperationRow) => void
}) {
  const { active_run_statuses: activeStatuses } = useOptions()
  const active = activeStatuses.includes(operation.status)
  const rows = tasks ?? []
  const byTask = rows.length > 0

  return (
    <details className="operation" onToggle={(event) => onToggle(operation, event.currentTarget.open)}>
      <summary>
        <div className="operation-head">
          <span className="tag">
            <span className="kind">{tagOf(operation)}</span>
            {/* the title is shortened when it does not fit, so the whole of it is the tooltip */}
            <span title={operation.title}>{operation.title}</span>
          </span>
          {/* the session whose action started it comes first: it is what the other columns lack */}
          <span className="mono muted">
            {operation.origin !== null && `${operation.origin} · `}
            {dateTime(operation.created_at)} · {duration(operation.updated_at - operation.created_at)}
          </span>
        </div>
        <Jobs jobs={jobsFor(operation, tasks, activeStatuses)} variant={active ? 'glass' : 'line'} stripes={active} />
      </summary>
      <div className="operation-tasks">
        {byTask ? (
          jobDefs(operation).map((def) => <TaskColumn key={def.stage} def={def} rows={rows.filter((task) => task.stage === def.stage)} />)
        ) : (
          <div className="operation-stage">
            <Kv rows={summaryRows(operation)} />
            {operation.detail.available === true && (
              <div className="row">
                <a className="btn btn-ghost" href={api.backupFileUrl(operation.id)} download>
                  <Download className="icon" />
                  Download
                </a>
              </div>
            )}
          </div>
        )}
      </div>
      {(operation.error !== null || active) && (
        <div className="operation-tasks">
          <div className="operation-stage">
            {operation.error !== null && <pre className="md">{operation.error}</pre>}
            {active && (
              <div className="row">
                <button className="btn btn-ghost" type="button" onClick={() => onCancel(operation)}>
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
 * What an operation that reports no tasks has to say: its status and whatever its kind counts. The
 * loaded model is on the strip above, so it is not repeated here.
 */
function summaryRows(operation: OperationRow): [string, string][] {
  const rows: [string, string][] = [['Status', operation.status.toLowerCase()]]
  if (operation.kind === 'document') rows.push(['Tasks', `${count(operation, 'tasks_done')}/${count(operation, 'tasks_total')}`])
  if (operation.kind === 'collection' && bulkKind(operation) === 'index_collection')
    rows.push(['Queued', `${count(operation, 'done')} of ${count(operation, 'total')}`])
  if (bulkKind(operation) === 'create_backup') rows.push(['Archived', `${count(operation, 'done')} of ${count(operation, 'total')} files`])
  if (typeof operation.detail.size === 'number') rows.push(['Size', bytes.format(operation.detail.size)])
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
