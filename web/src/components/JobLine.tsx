import { ACTIVE_JOB_STATUSES as ACTIVE, type JobRow, type Task } from '../api'

export function JobLine({ job: j, tasks, onToggle, onCancel }: { job: JobRow; tasks: Task[] | null; onToggle: () => void; onCancel: () => void }) {
  return (
    <>
      <tr>
        {/* an archived job is days old, so the day is part of when it started */}
        <td className="muted" title={j.id}>{new Date(j.created_at * 1000)[j.archived ? 'toLocaleString' : 'toLocaleTimeString']()}</td>
        <td>{j.title}</td>
        <td className={j.status === 'ERROR' ? 'error' : ''}>
          {j.status.toLowerCase()}
          {j.error && <pre className="error-detail">{j.error}</pre>}
        </td>
        <td>
          <Progress job={j} />
        </td>
        <td>
          {j.kind === 'document' && <button onClick={onToggle}>{tasks ? 'hide' : 'batches'}</button>}
          {ACTIVE.has(j.status) && <button onClick={onCancel}>cancel</button>}
        </td>
      </tr>
      {tasks && (
        <tr>
          <td></td>
          <td colSpan={4}>
            <table>
              <tbody>
                {tasks.map((w) => (
                  <tr key={w.id}>
                    <td className="muted">{w.stage} {w.seq}</td>
                    <td>{w.stage === 'convert' ? `pages ${w.page_start + 1}–${w.page_end}` : `part ${w.page_start}`}</td>
                    <td className={w.status === 'ERROR' ? 'error' : w.status === 'SUCCESS' ? '' : 'muted'}>
                      {w.status.toLowerCase()}
                      {w.error && <pre className="error-detail">{w.error}</pre>}
                    </td>
                    <td className="muted">
                      {w.result !== null && (w.stage === 'convert' ? `${w.result} OCR pages` : `${w.result} chunks`)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </td>
        </tr>
      )}
    </>
  )
}

// What one kind counts: micro-batches for a document, queued documents for a whole-collection
// job, and whether the backend process has the model loaded for a download. The rest count nothing.
function Progress({ job: j }: { job: JobRow }) {
  const count = (key: string) => (typeof j.detail[key] === 'number' ? j.detail[key] : 0)
  if (j.kind === 'document') {
    const [done, running, total] = [count('tasks_done'), count('tasks_running'), count('tasks_total')]
    // no batches at all: still planning while it runs; when it finished that way, the embed
    // found its cache and had nothing to compute
    if (total === 0) return <span className="muted">{ACTIVE.has(j.status) ? 'planning' : j.status === 'SUCCESS' ? 'already computed' : ''}</span>
    return (
      <>
        <Bar done={done} running={running} total={total} />
        <small className="muted">
          {' '}
          {done} done · {running} running · {total - done - running} queued
        </small>
      </>
    )
  }
  if (j.kind === 'collection' && typeof j.detail.total === 'number') {
    const skipped = count('skipped')
    return <small className="muted">{count('done')} of {count('total')} queued{skipped > 0 && ` · ${skipped} gone`}</small>
  }
  if (j.kind === 'download') {
    return <small className="muted">{j.detail.warm ? 'loaded' : 'not loaded here'}</small>
  }
  return null
}

function Bar({ done, running, total }: { done: number; running: number; total: number }) {
  const pct = (n: number) => `${(100 * n) / total}%`
  return (
    <span className="bar" title={`${done}/${total}`}>
      <span className="bar-done" style={{ width: pct(done) }} />
      <span className="bar-running" style={{ width: pct(running) }} />
    </span>
  )
}
