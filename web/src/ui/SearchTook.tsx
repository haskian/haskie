import { Fragment } from 'react'
import type { StepTiming } from '../api'

const ms = (value: number): string => `${value < 10 ? value.toFixed(1) : Math.round(value)} ms`

/** One line above the results: what came back, and how long it took. Blank but present before
 *  the first query, so the results do not jump down when it arrives. Hovering it opens, below it,
 *  how long each step of the search took on the server (`Server-Timing`). */
export function SearchTook({ counts, ms: took, steps = [] }: { counts: string; ms: number | null; steps?: StepTiming[] }) {
  return (
    <p className="mono muted search-took" tabIndex={steps.length > 0 ? 0 : undefined}>
      {took === null ? ' ' : `${counts} · ${took} ms`}
      {took !== null && steps.length > 0 && (
        <span className="hint hint-below" role="tooltip">
          <span className="label label-mono">Server · {ms(steps.reduce((sum, step) => sum + step.ms, 0))}</span>
          <span className="hint-rows">
            {steps.map((step) => (
              <Fragment key={step.step}>
                <span>{step.label}</span>
                <span className="muted">{step.step}</span>
                <span className="code">{ms(step.ms)}</span>
              </Fragment>
            ))}
          </span>
        </span>
      )}
    </p>
  )
}
