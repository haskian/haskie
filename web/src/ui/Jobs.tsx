import { Check, Clock, Settings, X } from 'lucide-react'
import type { CSSProperties } from 'react'
import { duration } from '../format'

export type JobState = 'done' | 'active' | 'todo' | 'error'

/** One bar of the strip: one job of an operation, or the operation itself where its kind has no
 *  jobs of its own. */
export interface JobBar {
  label: string
  done: number
  total: number
  state: JobState
  seconds?: number
  note?: string // what the bar has to say when it is not timed
  weight?: number // relative width; the jobs of an operation are not equally long
}

const ICONS = { done: Check, active: Settings, todo: Clock, error: X } as const

/** The progress strip in an operation summary: one bar per job, sized by weight, filled by
 *  progress. The `stage*` class names are the design system's (see `design/design.css`). */
export function Jobs({ jobs, variant, stripes }: { jobs: JobBar[]; variant: 'glass' | 'line'; stripes?: boolean }) {
  const className = ['stages', `stages-${variant}`, stripes ? 'stages-stripes' : null].filter(Boolean).join(' ')
  return (
    <div className={className}>
      {jobs.map((job) => (
        <Bar key={job.label} job={job} />
      ))}
    </div>
  )
}

function Bar({ job }: { job: JobBar }) {
  const Icon = ICONS[job.state]
  const done = job.state === 'done'
  // A finished job fills its bar whatever it counted: some kinds of operation count nothing at all.
  const progress = job.total > 0 ? job.done / job.total : done ? 1 : 0
  const style = { '--progress': progress, ...(job.weight === undefined ? {} : { '--weight': job.weight }) } as CSSProperties
  const className = done || job.state === 'active' ? `stage ${job.state}` : 'stage'
  const counted = job.total > 0
  // the note and the time share the right end: "cached · 3 sec"
  const said = [job.note, job.seconds === undefined ? undefined : duration(job.seconds)].filter(Boolean).join(' · ') || undefined

  return (
    <div className={className} style={style}>
      <span>{job.label}</span>
      <Icon className={job.state === 'active' ? 'icon spin' : 'icon'} />
      <span className="stage-bar" />
      {(counted || said !== undefined) && (
        <span className="stage-meta">
          {/* both ends always present, so the counts stay left and the time stays right */}
          <span>{counted ? `${job.done}/${job.total}` : ''}</span>
          <span>{said}</span>
        </span>
      )}
    </div>
  )
}
