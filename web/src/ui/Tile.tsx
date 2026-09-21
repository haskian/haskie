import { useRef, type ComponentType, type MouseEvent, type ReactNode } from 'react'

// Same rule as `design.js`: a hint that would run past the viewport flips to the tile's right edge.
const VIEWPORT_MARGIN = 16

/** A gallery square: icon, name, one line of detail, and the same detail again as a hover hint.
 *  `meta` is a date, set bottom right on the detail line, which goes mono beside it. */
export function Tile({
  icon: Icon,
  name,
  sub,
  meta,
  hint,
  onClick,
  pressed,
  add,
}: {
  icon: ComponentType<{ className?: string }>
  name: string
  sub: ReactNode // text, or a small `.glyph` icon and a number
  meta?: string
  hint: string
  onClick: () => void
  pressed?: boolean
  add?: boolean // the tile that makes a new thing rather than opening one
}) {
  const hintElement = useRef<HTMLSpanElement>(null)
  const flipIfClipped = (event: MouseEvent<HTMLButtonElement>) => {
    const element = hintElement.current
    if (!element) return
    const left = event.currentTarget.getBoundingClientRect().left
    element.classList.toggle('flip', left + element.offsetWidth > document.documentElement.clientWidth - VIEWPORT_MARGIN)
  }
  return (
    <button className={add ? 'tile tile-add' : 'tile'} type="button" aria-pressed={pressed} onClick={onClick} onMouseEnter={flipIfClipped}>
      <Icon className="icon" />
      <span className="tile-text">
        <span className="name">{name}</span>
        <span className="sub">{sub}</span>
        {meta !== undefined && <span className="meta">{meta}</span>}
      </span>
      <span className="hint" role="tooltip" ref={hintElement}>
        <strong>{name}</strong>
        <span className="sub">
          {sub}
          {meta !== undefined && ` · ${meta}`}
        </span>
        {hint}
      </span>
    </button>
  )
}
