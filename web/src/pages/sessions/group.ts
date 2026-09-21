import type { SessionSummary } from '../../api'
import { relative } from '../../format'
import type { RangeGroup } from '../../ui'

// ponytail: "active" is recent activity alone; a running job the session started would count
// too, but that needs the operations listing joined in. Add it when idle-with-a-job misleads.
const ACTIVE_WINDOW_SECONDS = 15 * 60

/** Two bands: the sessions seen in the last 15 minutes, then the rest; newest first in each,
 *  a session that did nothing yet last. */
export function groupByStatus(sessions: SessionSummary[], now: number): RangeGroup<SessionSummary>[] {
  const newestFirst = [...sessions].sort((a, b) => (b.last_at ?? -Infinity) - (a.last_at ?? -Infinity))
  const isActive = (one: SessionSummary): boolean => one.last_at !== null && now - one.last_at <= ACTIVE_WINDOW_SECONDS
  const active = newestFirst.filter(isActive)
  const idle = newestFirst.filter((one) => !isActive(one))
  return [
    { label: 'Active', items: active },
    { label: 'Idle', items: idle },
  ].filter((group) => group.items.length > 0)
}

/** The line under a session tile's id: how much it searches, and when it was last seen. */
export function tileSub(session: SessionSummary, now: number): string {
  const size = session.collections.length === 0 ? 'no collections' : `${session.collections.length} collections`
  return session.last_at === null ? `${size} · never used` : `${size} · ${relative(session.last_at, now)}`
}
