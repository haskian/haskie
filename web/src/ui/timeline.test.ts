import { describe, expect, test } from 'bun:test'
import type { StepTiming } from '../api'
import { timeline, type Stretch } from './timeline'

// The steps of a two-question excerpts search as the header carries them: the branches finish
// interleaved, Q2 before Q1 at first.
const plan: StepTiming = { step: 'plan', label: 'Embed the query', ms: 44 }
const q2Retrieve: StepTiming = { step: 'retrieve', label: 'LanceDB retrieval', ms: 8.3, branch: 'Q2' }
const q1Retrieve: StepTiming = { step: 'retrieve', label: 'LanceDB retrieval', ms: 8.1, branch: 'Q1' }
const q1Rerank: StepTiming = { step: 'rerank', label: 'Rerank', ms: 13253, branch: 'Q1' }
const q2Rerank: StepTiming = { step: 'rerank', label: 'Rerank', ms: 13872, branch: 'Q2' }
const fold: StepTiming = { step: 'fold', label: 'Take turns and fold passages', ms: 20 }

describe('timeline', () => {
  const cases: Array<{ name: string; steps: StepTiming[]; stretches: Stretch[]; ms: number }> = [
    { name: 'no steps, no time', steps: [], stretches: [], ms: 0 },
    {
      name: 'one run: every step in order, the total their sum',
      steps: [plan, fold],
      stretches: [
        { kind: 'step', step: plan },
        { kind: 'step', step: fold },
      ],
      ms: 64,
    },
    {
      name: 'branches side by side: one stretch, in name order, as long as the slowest',
      steps: [plan, q2Retrieve, q1Retrieve, q1Rerank, q2Rerank, fold],
      stretches: [
        { kind: 'step', step: plan },
        {
          kind: 'parallel',
          ms: 8.3 + 13872,
          branches: [
            { name: 'Q1', ms: 8.1 + 13253, steps: [q1Retrieve, q1Rerank] },
            { name: 'Q2', ms: 8.3 + 13872, steps: [q2Retrieve, q2Rerank] },
          ],
        },
        { kind: 'step', step: fold },
      ],
      ms: 44 + 8.3 + 13872 + 20,
    },
    {
      name: 'a shared step between branched ones splits them into two stretches',
      steps: [q1Retrieve, fold, q1Rerank],
      stretches: [
        { kind: 'parallel', ms: 8.1, branches: [{ name: 'Q1', ms: 8.1, steps: [q1Retrieve] }] },
        { kind: 'step', step: fold },
        { kind: 'parallel', ms: 13253, branches: [{ name: 'Q1', ms: 13253, steps: [q1Rerank] }] },
      ],
      ms: 8.1 + 20 + 13253,
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      const found = timeline(one.steps)
      expect(found.stretches).toEqual(one.stretches)
      expect(found.ms).toBeCloseTo(one.ms, 6)
    })
  }
})
