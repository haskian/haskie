import { describe, expect, test } from 'bun:test'
import type { JobKind, JobKindSummary, JobRow, Stage, StageJob, Task, WorkflowStatus } from '../api'
import type { StageState } from '../ui'
import { dayGroup, groupJobs, statusGroup, type GroupBy } from './operations/group'
import { endsOf, jobState, stageDefs, stageInfo, stagesFor, tagOf, taskState, taskText } from './operations/stages'

// 2019-01-19 14:32 local time: the instant the design page is drawn at.
const STARTED = new Date(2019, 0, 19, 14, 32).getTime() / 1000

// A real row of `/api/jobs/by-kind`: every field the backend sends, for a running import.
const JOB: JobRow = {
  id: 'import:renders/lamp.pdf:0194f2',
  kind: 'document',
  title: 'import renders/lamp.pdf',
  status: 'PENDING',
  created_at: STARTED,
  updated_at: STARTED + 38,
  error: null,
  origin: null,
  detail: { tasks_done: 9, tasks_running: 2, tasks_total: 22 },
  stages: [
    { stage: 'convert', job_id: 'import:renders/lamp.pdf:0194f2', status: 'SUCCESS', tasks_done: 3, tasks_running: 0, tasks_total: 3, seconds: 4 },
    { stage: 'embed', job_id: 'emb:renders/lamp.pdf:0194f2', status: 'PENDING', tasks_done: 6, tasks_running: 2, tasks_total: 19, seconds: null },
  ],
}
type StageJobPatch = Partial<StageJob> & { stage: Stage }
const stageJob = (patch: StageJobPatch): StageJob => ({ job_id: `${patch.stage}:renders/lamp.pdf:0194f2`, status: 'SUCCESS', tasks_done: 0, tasks_running: 0, tasks_total: 0, seconds: null, ...patch })

const job = (patch: Partial<JobRow>): JobRow => ({ ...JOB, ...patch })

// A real row of `/api/jobs/{id}/tasks`.
const TASK: Task = {
  id: 'import:renders/lamp.pdf:0194f2:convert:0',
  child_id: 'import:renders/lamp.pdf:0194f2:convert',
  stage: 'convert',
  seq: 0,
  page_start: 0,
  page_end: 10,
  status: 'SUCCESS',
  result: 2,
  error: null,
}

const task = (patch: Partial<Task> & { stage: Stage }): Task => ({ ...TASK, ...patch, id: `${TASK.id}:${patch.stage}:${patch.seq ?? 0}` })

const KINDS: JobKindSummary[] = [
  { kind: 'document', label: 'Document ingestion', active: 1 },
  { kind: 'collection', label: 'Collections', active: 0 },
  { kind: 'download', label: 'Model downloads', active: 0 },
  { kind: 'maintenance', label: 'Maintenance', active: 0 },
]

interface StagesCase {
  name: string
  job: JobRow
  tasks: Task[] | null
  expected: { label: string; done: number; total: number; state: StageState; note?: string; weight?: number; seconds?: number }[]
}

describe('stagesFor', () => {
  const cases: StagesCase[] = [
    {
      name: 'before its batches are read, each stage shows its own job: convert done, embed running',
      job: JOB,
      tasks: null,
      expected: [
        { label: 'Convert', done: 3, total: 3, state: 'done', weight: 1, note: undefined, seconds: 4 },
        { label: 'Embed', done: 6, total: 19, state: 'active', weight: 2.2, note: undefined, seconds: undefined },
      ],
    },
    {
      name: 'an index operation: embed the cache already held, then the write',
      job: job({
        id: 'idx-col:notes:renders/lamp.pdf:0194f2',
        title: 'notes / renders/lamp.pdf',
        status: 'SUCCESS',
        stages: [stageJob({ stage: 'embed' }), stageJob({ stage: 'index', tasks_done: 2, tasks_total: 2 })],
      }),
      tasks: null,
      expected: [
        { label: 'Embed', done: 0, total: 0, state: 'done', weight: 2.2, note: 'cached' },
        { label: 'Index', done: 2, total: 2, state: 'done', weight: 1, note: undefined },
      ],
    },
    {
      name: 'an embed with no parent on the page is one stage',
      job: job({ id: 'emb:renders/lamp.pdf:0194f2', title: 'embed renders/lamp.pdf', status: 'SUCCESS', stages: [stageJob({ stage: 'embed', tasks_done: 27, tasks_total: 27 })] }),
      tasks: null,
      expected: [{ label: 'Embed', done: 27, total: 27, state: 'done', weight: 2.2, note: undefined }],
    },
    {
      name: 'a failed operation: the stage job that failed reads as the error, the one still waiting as todo',
      job: job({
        status: 'ERROR',
        error: 'model gone',
        stages: [stageJob({ stage: 'convert', tasks_done: 3, tasks_total: 3 }), stageJob({ stage: 'embed', status: 'ERROR', tasks_done: 4, tasks_total: 19 })],
      }),
      tasks: null,
      expected: [
        { label: 'Convert', done: 3, total: 3, state: 'done', weight: 1, note: undefined },
        { label: 'Embed', done: 4, total: 19, state: 'error', weight: 2.2, note: undefined },
      ],
    },
    {
      name: 'a stage waiting for the one before it is todo',
      job: job({ stages: [stageJob({ stage: 'embed', status: 'PENDING', tasks_total: 19 }), stageJob({ stage: 'index', status: 'ENQUEUED' })] }),
      tasks: null,
      expected: [
        { label: 'Embed', done: 0, total: 19, state: 'active', weight: 2.2, note: undefined },
        { label: 'Index', done: 0, total: 0, state: 'active', weight: 1, note: undefined },
      ],
    },
    {
      name: 'a cancelled stage job is neither done nor running',
      job: job({ status: 'CANCELLED', stages: [stageJob({ stage: 'convert', status: 'CANCELLED', tasks_done: 1, tasks_total: 3 })] }),
      tasks: null,
      expected: [{ label: 'Convert', done: 1, total: 3, state: 'todo', weight: 1, note: undefined }],
    },
    {
      name: 'fetched batches decide each stage on their own',
      job: JOB,
      tasks: [
        task({ stage: 'convert', seq: 0 }),
        task({ stage: 'convert', seq: 1, page_start: 10, page_end: 14, result: 0 }),
        task({ stage: 'embed', seq: 0, page_start: 0, page_end: 1, status: 'SUCCESS', result: 14 }),
        task({ stage: 'embed', seq: 1, page_start: 1, page_end: 2, status: 'PENDING', result: null }),
      ],
      expected: [
        { label: 'Convert', done: 2, total: 2, state: 'done', weight: 1, seconds: 4 },
        { label: 'Embed', done: 1, total: 2, state: 'active', weight: 2.2, seconds: undefined },
      ],
    },
    {
      name: 'a stage whose batches failed reads as an error',
      job: job({ status: 'ERROR' }),
      tasks: [task({ stage: 'convert', seq: 0, status: 'ERROR', result: null, error: 'page 3: timeout' })],
      expected: [
        { label: 'Convert', done: 0, total: 1, state: 'error', weight: 1, seconds: 4 },
        { label: 'Embed', done: 6, total: 19, state: 'active', weight: 2.2, note: undefined, seconds: undefined },
      ],
    },
    {
      name: 'batches read for one stage leave the other on its job',
      job: job({ status: 'SUCCESS', stages: [stageJob({ stage: 'convert', tasks_done: 3, tasks_total: 3 }), stageJob({ stage: 'embed', tasks_done: 1, tasks_total: 1 })] }),
      tasks: [task({ stage: 'embed', seq: 0, page_start: 0, page_end: 1, status: 'SUCCESS', result: 14 })],
      expected: [
        { label: 'Convert', done: 3, total: 3, state: 'done', weight: 1, note: undefined },
        { label: 'Embed', done: 1, total: 1, state: 'done', weight: 2.2 },
      ],
    },
    {
      name: 'a collection job counts the documents it queued, and says how many it skipped',
      job: job({ kind: 'collection', title: 'index collection notes', status: 'PENDING', detail: { bulk: 'index_collection', done: 11, skipped: 1, total: 24 } }),
      tasks: null,
      expected: [{ label: 'Queue', done: 11, total: 24, state: 'active', note: '1 skipped' }],
    },
    {
      name: 'a finished collection job with no progress event counts nothing and skipped nothing',
      job: job({ kind: 'collection', title: 'index collection notes', status: 'SUCCESS', detail: {} }),
      tasks: null,
      expected: [{ label: 'Queue', done: 0, total: 0, state: 'done', note: undefined, seconds: 38 }],
    },
    {
      name: 'a document delete is a delete, not a queue',
      job: job({ kind: 'collection', title: 'delete document gone.pdf', status: 'SUCCESS', detail: { bulk: 'delete_document' } }),
      tasks: null,
      expected: [{ label: 'Delete', done: 0, total: 0, state: 'done', note: undefined, seconds: 38 }],
    },
    {
      name: 'a download is one task, done 1/1 and timed, and says whether the model is loaded',
      job: job({ kind: 'download', title: 'download embedding BAAI/bge-small-en-v1.5', status: 'SUCCESS', detail: { warm: true } }),
      tasks: null,
      expected: [{ label: 'Download', done: 1, total: 1, state: 'done', note: 'loaded', seconds: 38 }],
    },
    {
      name: 'a model the process never loaded says so',
      job: job({ kind: 'download', title: 'download embedding BAAI/bge-small-en-v1.5', status: 'SUCCESS', detail: { warm: false } }),
      tasks: null,
      expected: [{ label: 'Download', done: 1, total: 1, state: 'done', note: 'not loaded', seconds: 38 }],
    },
    {
      name: 'a maintenance run that failed shows an error stage',
      job: job({ kind: 'maintenance', title: 'maintain notes', status: 'ERROR', error: 'lance: commit conflict', detail: {} }),
      tasks: null,
      expected: [{ label: 'Maintenance', done: 0, total: 1, state: 'error', seconds: 38 }],
    },
    {
      name: 'a cancelled maintenance run is neither done nor running',
      job: job({ kind: 'maintenance', title: 'maintain notes', status: 'CANCELLED', detail: {} }),
      tasks: null,
      expected: [{ label: 'Maintenance', done: 0, total: 1, state: 'todo', seconds: undefined }],
    },
  ]

  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(stagesFor(testCase.job, testCase.tasks)).toEqual(testCase.expected)
    })
  }
})

describe('jobState', () => {
  const cases: { name: string; status: WorkflowStatus; expected: StageState }[] = [
    { name: 'success is done', status: 'SUCCESS', expected: 'done' },
    { name: 'error is an error', status: 'ERROR', expected: 'error' },
    { name: 'enqueued is active', status: 'ENQUEUED', expected: 'active' },
    { name: 'pending is active', status: 'PENDING', expected: 'active' },
    { name: 'cancelled is neither', status: 'CANCELLED', expected: 'todo' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(jobState(testCase.status)).toBe(testCase.expected)
    })
  }
})

describe('taskText', () => {
  const cases: { name: string; task: Task; expected: string }[] = [
    { name: 'a conversion names its pages, counted from one', task: task({ stage: 'convert', page_start: 0, page_end: 10 }), expected: 'pages 1–10' },
    { name: 'an embed names its one part', task: task({ stage: 'embed', page_start: 3, page_end: 4 }), expected: 'part 3' },
    { name: 'an index write names the range it wrote', task: task({ stage: 'index', page_start: 0, page_end: 50 }), expected: 'parts 0–50' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(taskText(testCase.task)).toBe(testCase.expected)
    })
  }
})

describe('taskState', () => {
  const cases: { name: string; status: WorkflowStatus; expected: 'done' | 'error' | 'todo' }[] = [
    { name: 'success is done', status: 'SUCCESS', expected: 'done' },
    { name: 'error is an error', status: 'ERROR', expected: 'error' },
    { name: 'anything unfinished is todo', status: 'PENDING', expected: 'todo' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(taskState(task({ stage: 'embed', status: testCase.status }))).toBe(testCase.expected)
    })
  }
})

describe('stageInfo', () => {
  const cases: { name: string; stage: Stage; rows: Task[]; expected: string }[] = [
    { name: 'no batches says nothing', stage: 'convert', rows: [], expected: '' },
    { name: 'batches with no result yet say nothing', stage: 'embed', rows: [task({ stage: 'embed', status: 'PENDING', result: null })], expected: '' },
    {
      name: 'a conversion sums the pages that needed OCR',
      stage: 'convert',
      rows: [task({ stage: 'convert', seq: 0, result: 2 }), task({ stage: 'convert', seq: 1, result: 3 })],
      expected: '5 OCR pages',
    },
    {
      name: 'an embed sums its chunks, ignoring the batches still running',
      stage: 'embed',
      rows: [task({ stage: 'embed', seq: 0, result: 14 }), task({ stage: 'embed', seq: 1, status: 'PENDING', result: null })],
      expected: '14 chunks',
    },
    { name: 'an index write sums its chunks', stage: 'index', rows: [task({ stage: 'index', result: 31 })], expected: '31 chunks' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(stageInfo(testCase.stage, testCase.rows)).toBe(testCase.expected)
    })
  }
})

describe('statusGroup', () => {
  const cases: { name: string; status: WorkflowStatus; expected: string }[] = [
    { name: 'enqueued is running', status: 'ENQUEUED', expected: 'Running' },
    { name: 'pending is running', status: 'PENDING', expected: 'Running' },
    { name: 'success is completed', status: 'SUCCESS', expected: 'Completed' },
    { name: 'error is failed', status: 'ERROR', expected: 'Failed' },
    { name: 'cancelled keeps its own word', status: 'CANCELLED', expected: 'Cancelled' },
    { name: 'a status the page does not know keeps its raw name', status: 'MAX_RECOVERY_ATTEMPTS_EXCEEDED', expected: 'MAX_RECOVERY_ATTEMPTS_EXCEEDED' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(statusGroup(job({ status: testCase.status }))).toBe(testCase.expected)
    })
  }
})

test('dayGroup is the date without the time', () => {
  expect(dayGroup(JOB)).toBe('Sat 19 Jan')
})

describe('groupJobs', () => {
  const running = job({ id: 'a', status: 'PENDING', created_at: STARTED })
  const finished = job({ id: 'b', status: 'SUCCESS', created_at: STARTED - 600 })
  const failed = job({ id: 'c', status: 'ERROR', created_at: STARTED - 1200 })
  const yesterday = job({ id: 'd', kind: 'collection' as JobKind, status: 'SUCCESS', created_at: STARTED - 24 * 3600 })
  const newest = [running, finished, failed, yesterday]

  const cases: { name: string; jobs: JobRow[]; by: GroupBy; expected: { key: string; ids: string[] }[] }[] = [
    { name: 'no jobs, no sections', jobs: [], by: 'status', expected: [] },
    {
      name: 'by status, running first and failed after completed',
      jobs: newest,
      by: 'status',
      expected: [
        { key: 'Running', ids: ['a'] },
        { key: 'Completed', ids: ['b', 'd'] },
        { key: 'Failed', ids: ['c'] },
      ],
    },
    {
      name: 'an unknown status sorts after the ones the page knows',
      jobs: [job({ id: 'x', status: 'MAX_RECOVERY_ATTEMPTS_EXCEEDED' }), finished],
      by: 'status',
      expected: [
        { key: 'Completed', ids: ['b'] },
        { key: 'MAX_RECOVERY_ATTEMPTS_EXCEEDED', ids: ['x'] },
      ],
    },
    {
      name: 'by kind, in the order the backend lists the kinds, empty kinds dropped',
      jobs: newest,
      by: 'kind',
      expected: [
        { key: 'Document ingestion', ids: ['a', 'b', 'c'] },
        { key: 'Collections', ids: ['d'] },
      ],
    },
    {
      name: 'by day, newest day first',
      jobs: newest,
      by: 'day',
      expected: [
        { key: 'Sat 19 Jan', ids: ['a', 'b', 'c'] },
        { key: 'Fri 18 Jan', ids: ['d'] },
      ],
    },
  ]

  for (const testCase of cases) {
    test(testCase.name, () => {
      const groups = groupJobs(testCase.jobs, testCase.by, KINDS)
      expect(groups.map((group) => ({ key: group.key, ids: group.jobs.map((row) => row.id) }))).toEqual(testCase.expected)
    })
  }
})

describe('stageDefs', () => {
  const cases: { name: string; stages: Stage[]; expected: Stage[] }[] = [
    { name: 'an import: convert, then embed', stages: ['convert', 'embed'], expected: ['convert', 'embed'] },
    { name: 'an index: embed, then index', stages: ['embed', 'index'], expected: ['embed', 'index'] },
    { name: 'an embed on its own', stages: ['embed'], expected: ['embed'] },
    { name: 'no stages, no bars', stages: [], expected: [] },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(stageDefs(job({ stages: one.stages.map((stage) => stageJob({ stage })) })).map((def) => def.stage)).toEqual(one.expected)
    })
  }
})

describe('tagOf', () => {
  const cases: { name: string; row: JobRow; expected: string }[] = [
    { name: 'an operation that converts is an import', row: JOB, expected: 'Import' },
    { name: 'one that writes a collection is an index', row: job({ stages: [stageJob({ stage: 'embed' }), stageJob({ stage: 'index' })] }), expected: 'Index' },
    { name: 'an embed on its own', row: job({ stages: [stageJob({ stage: 'embed' })] }), expected: 'Embed' },
    { name: 'another kind keeps its word', row: job({ kind: 'maintenance', stages: [] }), expected: 'Maintain' },
    { name: 'a collection index', row: job({ kind: 'collection', stages: [], detail: { bulk: 'index_collection' } }), expected: 'Index' },
    { name: 'a collection delete', row: job({ kind: 'collection', stages: [], detail: { bulk: 'delete_collection' } }), expected: 'Delete' },
    { name: 'a collection row that does not say is an index', row: job({ kind: 'collection', stages: [], detail: {} }), expected: 'Index' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(tagOf(one.row)).toBe(one.expected)
    })
  }
})

describe('endsOf', () => {
  const rows = (n: number): number[] => Array.from({ length: n }, (_, index) => index)
  const cases: { name: string; rows: number[]; expected: { head: number[]; hidden: number; tail: number[] } }[] = [
    { name: 'nothing', rows: [], expected: { head: [], hidden: 0, tail: [] } },
    { name: 'six or fewer are all listed', rows: rows(6), expected: { head: rows(6), hidden: 0, tail: [] } },
    { name: 'seven: three, one counted, three', rows: rows(7), expected: { head: [0, 1, 2], hidden: 1, tail: [4, 5, 6] } },
    { name: 'many: the ends, the middle counted', rows: rows(27), expected: { head: [0, 1, 2], hidden: 21, tail: [24, 25, 26] } },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(endsOf(one.rows)).toEqual(one.expected)
    })
  }
})
