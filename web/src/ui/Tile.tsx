import { useRef, type ComponentType, type MouseEvent, type ReactNode } from 'react'

// Same rule as `design.js`: a hint that would run past the viewport flips to the tile's right edge.
const VIEWPORT_MARGIN = 16

/** A gallery square: icon, name and one line of detail. On hover, `hint` shows below the same
 *  name and detail again, while `description` shows alone, and nothing shows when it is empty.
 *  `meta` is a date, set bottom right on the detail line, which goes mono beside it. `cover` is
 *  the URL of a picture shown behind it, fading out where the text sits. */
export function Tile({
  icon: Icon,
  name,
  sub,
  meta,
  hint,
  description,
  cover,
  onClick,
  pressed,
  add,
}: {
  icon: ComponentType<{ className?: string }>
  name: string
  sub: ReactNode // text, or a small `.glyph` icon and a number
  meta?: string
  hint?: string
  description?: string
  cover?: string
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
  const tip =
    hint === undefined ? (
      description
    ) : (
      <>
        <strong>{name}</strong>
        <span className="sub">
          {sub}
          {meta !== undefined && ` · ${meta}`}
        </span>
        {hint}
      </>
    )
  return (
    <button className={add ? 'tile tile-add' : 'tile'} type="button" aria-pressed={pressed} onClick={onClick} onMouseEnter={flipIfClipped}>
      {cover !== undefined && <img className="tile-cover" src={cover} alt="" loading="lazy" />}
      <Icon className="icon" />
      <span className="tile-text">
        <span className="name">{name}</span>
        <span className="sub">{sub}</span>
        {meta !== undefined && <span className="meta">{meta}</span>}
      </span>
      {tip && (
        <span className="hint" role="tooltip" ref={hintElement}>
          {tip}
        </span>
      )}
    </button>
  )
}
