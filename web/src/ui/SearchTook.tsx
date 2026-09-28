import { Fragment } from 'react'
import type { StepTiming } from '../api'
import { timeline } from './timeline'

const ms = (value: number): string => `${value < 10 ? value.toFixed(1) : Math.round(value)} ms`

function Step({ step }: { step: StepTiming }) {
  return (
    <>
      <span>{step.label}</span>
      <span className="muted">{step.step}</span>
      <span className="code">{ms(step.ms)}</span>
    </>
  )
}

/** One line above the results: what came back, and how long it took. Blank but present before
 *  the first query, so the results do not jump down when it arrives. Hovering it opens, below it,
 *  how long each step of the search took on the server (`Server-Timing`): the steps every run
 *  shared, and the branches that ran side by side, each under a rule. */
export function SearchTook({ counts, ms: took, steps = [] }: { counts: string; ms: number | null; steps?: StepTiming[] }) {
  const { stretches, ms: server } = timeline(steps)
  return (
    <p className="mono muted search-took" tabIndex={steps.length > 0 ? 0 : undefined}>
      {took === null ? ' ' : `${counts} · ${took} ms`}
      {took !== null && steps.length > 0 && (
        <span className="hint hint-below" role="tooltip">
          <span className="label label-mono">Server · {ms(server)}</span>
          <span className="hint-rows">
            {stretches.map((stretch, at) =>
              stretch.kind === 'step' ? (
                <Step key={at} step={stretch.step} />
              ) : (
                <span key={at} className="hint-parallel">
                  <span className="label label-mono">In parallel · {stretch.branches.length}</span>
                  <span className="muted">slowest</span>
                  <span className="code">{ms(stretch.ms)}</span>
                  {stretch.branches.map((branch) => (
                    <Fragment key={branch.name}>
                      <span className="label label-mono">{branch.name}</span>
                      <span />
                      <span className="code muted">{ms(branch.ms)}</span>
                      {branch.steps.map((step, index) => (
                        <Step key={index} step={step} />
                      ))}
                    </Fragment>
                  ))}
                </span>
              ),
            )}
          </span>
        </span>
      )}
    </p>
  )
}
