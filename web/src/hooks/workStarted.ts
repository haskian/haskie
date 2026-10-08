// Background work a page just started: the status bar reads it at once rather than on its next
// poll, since starting work is what loads a knowledge model (the describer) on the server.
const bus = new EventTarget()

export function workStarted() {
  bus.dispatchEvent(new Event('started'))
}

export function onWorkStarted(listener: () => void): () => void {
  bus.addEventListener('started', listener)
  return () => bus.removeEventListener('started', listener)
}
