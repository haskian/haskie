import { Check, Clock, Settings, X } from 'lucide-react'
import type { CSSProperties } from 'react'
import { duration } from '../format'

export type StageState = 'done' | 'active' | 'todo' | 'error'

export interface StageRow {
  label: string
  done: number
  total: number
  state: StageState
  seconds?: number
  note?: string // what the stage has to say when it is not timed
  weight?: number // relative width; stages of a job are not equally long
}

const ICONS = { done: Check, active: Settings, todo: Clock, error: X } as const

/** The progress strip in a job summary: one bar per stage, sized by weight, filled by progress. */
export function Stages({ stages, variant, stripes }: { stages: StageRow[]; variant: 'glass' | 'line'; stripes?: boolean }) {
  const className = ['stages', `stages-${variant}`, stripes ? 'stages-stripes' : null].filter(Boolean).join(' ')
  return (
    <div className={className}>
      {stages.map((stage) => (
        <Stage key={stage.label} stage={stage} />
      ))}
    </div>
  )
}

function Stage({ stage }: { stage: StageRow }) {
  const Icon = ICONS[stage.state]
  const done = stage.state === 'done'
  // A finished stage fills its bar whatever it counted: some kinds of job count nothing at all.
  const progress = stage.total > 0 ? stage.done / stage.total : done ? 1 : 0
  const style = { '--progress': progress, ...(stage.weight === undefined ? {} : { '--weight': stage.weight }) } as CSSProperties
  const className = done || stage.state === 'active' ? `stage ${stage.state}` : 'stage'
  const counted = stage.total > 0
  // the note and the time share the right end: "cached · 3 sec"
  const said = [stage.note, stage.seconds === undefined ? undefined : duration(stage.seconds)].filter(Boolean).join(' · ') || undefined

  return (
    <div className={className} style={style}>
      <span>{stage.label}</span>
      <Icon className={stage.state === 'active' ? 'icon spin' : 'icon'} />
      <span className="stage-bar" />
      {(counted || said !== undefined) && (
        <span className="stage-meta">
          {/* both ends always present, so the counts stay left and the time stays right */}
          <span>{counted ? `${stage.done}/${stage.total}` : ''}</span>
          <span>{said}</span>
        </span>
      )}
    </div>
  )
}
