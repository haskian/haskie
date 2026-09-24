import { describe, expect, test } from 'bun:test'
import { dateTime, duration, relative } from './format'

// Local time in, local time out: the expectation holds in any time zone the test runs in.
const at = (year: number, month: number, day: number, hour: number, minute: number): number =>
  new Date(year, month - 1, day, hour, minute).getTime() / 1000

// `bytes` is not tested. It formats in the reader's own locale, so every expectation
// would assert the runtime's locale data rather than anything this file decides.

describe('dateTime', () => {
  const cases: Array<{ name: string; value: number; expected: string }> = [
    { name: 'afternoon on a Saturday', value: at(2019, 1, 19, 14, 32), expected: 'Sat 19 Jan · 14:32' },
    { name: 'pads the hour before ten', value: at(2019, 1, 19, 9, 5), expected: 'Sat 19 Jan · 09:05' },
    { name: 'midnight is 00, not 24', value: at(2019, 1, 19, 0, 0), expected: 'Sat 19 Jan · 00:00' },
    { name: 'a single-digit day is not padded', value: at(2019, 2, 3, 7, 45), expected: 'Sun 3 Feb · 07:45' },
    { name: 'the last minute of the year', value: at(2019, 12, 31, 23, 59), expected: 'Tue 31 Dec · 23:59' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(dateTime(testCase.value)).toBe(testCase.expected)
    })
  }
})

describe('duration', () => {
  const cases: Array<{ name: string; value: number; expected: string }> = [
    { name: 'zero', value: 0, expected: '0 sec' },
    { name: 'a negative span clamps to zero', value: -5, expected: '0 sec' },
    { name: 'seconds', value: 38, expected: '38 sec' },
    { name: 'fractional seconds round', value: 38.6, expected: '39 sec' },
    { name: 'just under a minute stays in seconds', value: 59, expected: '59 sec' },
    { name: 'exactly a minute switches unit', value: 60, expected: '1 min' },
    { name: 'minutes', value: 120, expected: '2 min' },
    { name: 'just under an hour stays in minutes', value: 3599, expected: '60 min' },
    { name: 'exactly an hour drops the zero minutes', value: 3600, expected: '1 h' },
    { name: 'hours and minutes', value: 3840, expected: '1 h 4 min' },
    { name: 'many hours', value: 7260, expected: '2 h 1 min' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(duration(testCase.value)).toBe(testCase.expected)
    })
  }
})

describe('relative', () => {
  const now = at(2019, 1, 19, 14, 32)
  const cases: Array<{ name: string; value: number; expected: string }> = [
    { name: 'the same instant', value: now, expected: '0 sec ago' },
    { name: 'seconds ago', value: now - 38, expected: '38 sec ago' },
    { name: 'minutes ago', value: now - 120, expected: '2 min ago' },
    { name: 'hours ago', value: now - 3840, expected: '1 h 4 min ago' },
    { name: 'a future timestamp reads forward', value: now + 120, expected: 'in 2 min' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(relative(testCase.value, now)).toBe(testCase.expected)
    })
  }
})
