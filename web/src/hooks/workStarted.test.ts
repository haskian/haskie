import { expect, test } from 'bun:test'
import { onWorkStarted, workStarted } from './workStarted'

test('a listener hears every start until it stops listening', () => {
  let heard = 0
  const stop = onWorkStarted(() => (heard += 1))
  workStarted()
  workStarted()
  stop()
  workStarted()
  expect(heard).toBe(2)
})
