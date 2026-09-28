import type { StepTiming } from '../api'

/** One run of steps among several side by side: its name (`Q1`), its steps in order, and their sum. */
export interface Branch {
  name: string
  ms: number
  steps: StepTiming[]
}

/** A stretch of the server's time: one step every run shared, or several branches side by side,
 *  which take as long as the slowest of them. */
export type Stretch = { kind: 'step'; step: StepTiming } | { kind: 'parallel'; ms: number; branches: Branch[] }

/** The steps as the server ran them, and how long that took. Consecutive steps that name a branch
 *  form one parallel stretch; the branches are in name order, since they finish interleaved. The
 *  total adds the slowest branch of a stretch, not all of them: they ran at once. */
export function timeline(steps: StepTiming[]): { stretches: Stretch[]; ms: number } {
  const stretches: Stretch[] = []
  for (const step of steps) {
    if (step.branch === undefined) {
      stretches.push({ kind: 'step', step })
      continue
    }
    let stretch = stretches.at(-1)
    if (stretch?.kind !== 'parallel') {
      stretch = { kind: 'parallel', ms: 0, branches: [] }
      stretches.push(stretch)
    }
    let branch = stretch.branches.find((one) => one.name === step.branch)
    if (branch === undefined) {
      branch = { name: step.branch, ms: 0, steps: [] }
      stretch.branches.push(branch)
    }
    branch.steps.push(step)
    branch.ms += step.ms
    stretch.ms = Math.max(stretch.ms, branch.ms)
  }
  for (const stretch of stretches) {
    if (stretch.kind === 'parallel') stretch.branches.sort((a, b) => a.name.localeCompare(b.name))
  }
  const ms = stretches.reduce((sum, stretch) => sum + (stretch.kind === 'step' ? stretch.step.ms : stretch.ms), 0)
  return { stretches, ms }
}
