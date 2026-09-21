// Number and time formats shared by the views, built once: a formatter is expensive to create and
// the listings format a value per row.
export const bytes = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  style: 'unit',
  unit: 'byte',
  unitDisplay: 'narrow',
})

const SECONDS_PER_MINUTE = 60
const SECONDS_PER_HOUR = 3600

// `en-GB` fixes the field shapes ("Sat", "Jan", 24-hour clock); the parts are assembled by hand so
// the separator matches the design pages exactly.
const dateTimeParts = new Intl.DateTimeFormat('en-GB', {
  weekday: 'short',
  day: 'numeric',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})

const partsOf = (unixSeconds: number): Map<string, string> =>
  new Map(dateTimeParts.formatToParts(new Date(unixSeconds * 1000)).map((part) => [part.type, part.value]))

/** The day a timestamp falls on, in the reader's own time zone: "Sat 19 Jan". */
export function day(unixSeconds: number): string {
  const parts = partsOf(unixSeconds)
  return `${parts.get('weekday')} ${parts.get('day')} ${parts.get('month')}`
}

/** A job or import timestamp, in the reader's own time zone: "Sat 19 Jan · 14:32". */
export function dateTime(unixSeconds: number): string {
  const parts = partsOf(unixSeconds)
  return `${day(unixSeconds)} · ${parts.get('hour')}:${parts.get('minute')}`
}

/** An elapsed span at one unit of precision: "38 sec", "2 min", "1 h 4 min", "2 h". */
export function duration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  if (total < SECONDS_PER_MINUTE) return `${total} sec`
  if (total < SECONDS_PER_HOUR) return `${Math.round(total / SECONDS_PER_MINUTE)} min`
  const hours = Math.floor(total / SECONDS_PER_HOUR)
  const minutes = Math.round((total % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE)
  return minutes === 0 ? `${hours} h` : `${hours} h ${minutes} min`
}

// `now` is an argument rather than a read of the clock so a caller can render a whole list against
// one instant, and so the result is testable.
/** How long ago a timestamp was: "2 min ago". A future timestamp reads "in 2 min". */
export function relative(unixSeconds: number, now: number): string {
  const elapsed = now - unixSeconds
  return elapsed < 0 ? `in ${duration(-elapsed)}` : `${duration(elapsed)} ago`
}

// What a failed request says on screen: the message alone, not `Error: ` in front of it.
export const errorText = (cause: unknown): string => (cause instanceof Error ? cause.message : String(cause))

/** What a reader typed, ready to match with: trimmed and folded to lower case. */
export const needleOf = (search: string): string => search.trim().toLowerCase()

// The listings filter client-side over the rows already loaded, so every page matches the same way:
// an empty query keeps everything, and any field containing the needle is a hit.
/** Whether a row matches a `needleOf` query against the fields a row shows. */
export const matchesText = (needle: string, ...fields: string[]): boolean =>
  needle === '' || fields.some((field) => field.toLowerCase().includes(needle))
