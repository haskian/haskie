import { useEffect } from 'react'

// Re-run `refresh` every `ms` while `active` and the tab is visible; used by every view that
// watches background work. A hidden tab polls nothing: the answer is not on screen, and the
// first refresh after `visibilitychange` catches up on whatever happened meanwhile.
export function usePoll(active: boolean, refresh: () => void, ms = 1500) {
  useEffect(() => {
    if (!active) return
    let timer: ReturnType<typeof setInterval> | undefined
    const stop = () => {
      clearInterval(timer)
      timer = undefined
    }
    const sync = () => {
      if (document.visibilityState !== 'visible') return stop()
      if (timer === undefined) timer = setInterval(refresh, ms)
    }
    sync()
    document.addEventListener('visibilitychange', sync)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', sync)
    }
  }, [active, refresh, ms])
}
