import { useEffect, useRef, useState } from 'react'
import { api, type Head, type Preview } from '../api'
import { errorText } from '../format'
import { anchorIndex, breadcrumb, type Anchor } from './anchor'
import { Skeleton } from './Skeleton'

// The server numbers heading ids by position (`render.to_html`), so the nth entry of the table of
// contents links to `h-n`. Nothing here counts renders: the position in the document is fixed.
const anchorId = (index: number): string => `h-${index}`

interface Position {
  heading: number | null
  page: number
}

/** The index of the last element whose top is at or above `top`, by binary search; null when none is. */
function lastAtOrAbove(elements: ArrayLike<HTMLElement>, top: number): number | null {
  let low = 0
  let high = elements.length - 1
  let found: number | null = null
  while (low <= high) {
    const mid = (low + high) >> 1
    if (elements[mid].getBoundingClientRect().top <= top) {
      found = mid
      low = mid + 1
    } else {
      high = mid - 1
    }
  }
  return found
}

/** What sits at the top of the pane: the page (from 1) and the last heading at or above it. */
function atTop(pane: HTMLElement): Position {
  const top = pane.getBoundingClientRect().top + pane.clientTop + 1 // clientTop: the inset is a border
  const headings = pane.querySelectorAll<HTMLElement>('.markdown [id^="h-"]')
  const at = lastAtOrAbove(headings, top)
  const page = lastAtOrAbove(pane.querySelectorAll<HTMLElement>('.markdown > div'), top)
  return { heading: at === null ? null : Number(headings[at].id.slice(2)), page: page === null ? 1 : page + 1 }
}

/** Bring a heading to the top of its pane without scrolling anything outside the pane. */
function scrollPaneTo(heading: HTMLElement): void {
  const pane = heading.closest('.pane-body')
  if (pane === null) return
  pane.scrollTop += heading.getBoundingClientRect().top - pane.getBoundingClientRect().top - pane.clientTop
}

const PANE_HEADS: Record<Preview['kind'], string> = { pdf: 'PDF', html: 'HTML', image: 'Image', text: 'Text' }

/**
 * The two-pane document view: the original on the left, the rendered markdown on the right. The
 * markdown arrives page by page over NDJSON, so a long document shows its first page at once.
 */
export function DocumentPanes({
  doc,
  preview,
  full = false,
  anchor,
  shown = true,
}: {
  doc: string
  preview: Preview | null
  full?: boolean
  anchor?: Anchor // opened from a search result: the markdown starts at its heading
  shown?: boolean // false while the panes sit in a hidden tab, where nothing can scroll
}) {
  const [head, setHead] = useState<Head | null>(null)
  const [pages, setPages] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loadedSource, setLoadedSource] = useState<string | null>(null) // the URL the frame or image has shown
  // The source frame mounts the first time the panes are on screen, never in a hidden tab: a PDF
  // viewer that starts life display: none stays blank in Chrome. Derived during render, as React
  // wants state that follows a prop to be.
  const [seen, setSeen] = useState(shown)
  if (shown && !seen) setSeen(true)
  const scrolled = useRef<number | null>(null) // the anchor already jumped to
  const [current, setCurrent] = useState<Position>({ heading: null, page: 1 }) // what the pane is scrolled to
  const frame = useRef(0)

  // One read per frame, however fast the pane scrolls.
  const onScroll = (event: { currentTarget: HTMLElement }) => {
    const pane = event.currentTarget
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => setCurrent(atTop(pane)))
  }

  useEffect(() => {
    let live = true
    const read = async () => {
      // cleared here rather than in the effect body: this is the start of the new request, and
      // the old document stays on screen until its replacement begins to arrive
      setHead(null)
      setPages([])
      setError(null)
      for await (const frame of api.markdown(doc, full)) {
        if (!live) return // the document changed while this one was still arriving
        if (frame.kind === 'head') setHead(frame)
        else setPages((shown) => [...shown, frame.html])
      }
    }
    read().catch((cause: unknown) => live && setError(errorText(cause)))
    return () => {
      live = false
    }
  }, [doc, full])

  // The pages stream in, so the heading may not be on screen yet: each new page tries again,
  // as does the tab the panes are in becoming visible.
  const target = head !== null && anchor !== undefined ? anchorIndex(head.toc, anchor) : null
  const arrived = pages.length
  useEffect(() => {
    if (!shown || target === null || scrolled.current === target) return
    const heading = document.getElementById(anchorId(target))
    if (heading === null) return
    scrollPaneTo(heading)
    scrolled.current = target
  }, [shown, target, arrived])

  // The row may predate the preview build (it is built on first open), so its `preview` can be
  // null while the head frame, streamed after the build, names the kind. Until either does, the
  // skeleton holds the room.
  const source = head?.preview ?? preview
  const previewUrl = api.previewUrl(doc)
  const sourceReady = loadedSource === previewUrl // a new document starts over without an effect
  const loading = sourceReady ? undefined : 'loading'
  const trail = head !== null && head.toc.length > 0 ? breadcrumb(head.toc, current.heading ?? 0) : []
  const crumb = [head !== null && head.pages > 1 ? `p. ${current.page}/${head.pages}` : '', trail.join(' › ')].filter(Boolean).join(' · ')

  return (
    <div className="split">
      <section className="pane">
        <PaneHead label={source ? PANE_HEADS[source.kind] : 'Source'} />
        <div className="pane-body">
          {/* the skeleton holds the room; the frame loads out of sight, then takes it */}
          {source?.kind !== 'text' && !sourceReady && <Skeleton />}
          {seen && source?.kind === 'pdf' && <iframe title="source" src={previewUrl} className={loading} onLoad={() => setLoadedSource(previewUrl)} />}
          {seen && source?.kind === 'html' && <iframe title="source" src={previewUrl} sandbox="" className={loading} onLoad={() => setLoadedSource(previewUrl)} />}
          {seen && source?.kind === 'image' && <img alt={doc} src={previewUrl} className={loading} onLoad={() => setLoadedSource(previewUrl)} />}
          {source?.kind === 'text' && <TextPane url={previewUrl} />}
        </div>
      </section>
      <section className="pane">
        <PaneHead label="Markdown" crumb={crumb} />
        <div className="pane-body" onScroll={onScroll}>
          {error !== null && <p className="muted">{error}</p>}
          {head === null && error === null && <Skeleton />}
          {head !== null && head.toc.length > 0 && (
            <details className="toc" open>
              <summary className="mono muted">Contents</summary>
              {head.toc.map((heading, index) => (
                <div key={anchorId(index)} style={{ paddingLeft: `${(heading.level - 1) * 12}px` }}>
                  {/* the hash is the router's, so a heading link scrolls instead of navigating */}
                  <a
                    href={`#${anchorId(index)}`}
                    onClick={(event) => {
                      event.preventDefault()
                      const heading = document.getElementById(anchorId(index))
                      if (heading !== null) scrollPaneTo(heading)
                    }}
                  >
                    {heading.text}
                  </a>
                </div>
              ))}
            </details>
          )}
          {/* The HTML is rendered by the server, which strips raw HTML first (see render.py), so
              there is nothing here for a document to inject. One element per page, appended as
              each frame arrives. */}
          <div className="markdown">
            {pages.map((html, index) => (
              <div key={index} dangerouslySetInnerHTML={{ __html: html }} />
            ))}
          </div>
          {head !== null && pages.length < head.pages && (
            <p className="muted">
              {pages.length} of {head.pages} pages…
            </p>
          )}
        </div>
      </section>
    </div>
  )
}

/** Both panes wear the same head: the kind on the left, where the pane is scrolled to on the right. */
function PaneHead({ label, crumb = '' }: { label: string; crumb?: string }) {
  return (
    <div className="pane-head mono muted">
      <span>{label}</span>
      <span className="crumb" title={crumb}>
        {crumb}
      </span>
    </div>
  )
}

function TextPane({ url }: { url: string }) {
  const [text, setText] = useState('')
  useEffect(() => {
    let live = true
    fetch(url)
      .then((response) => response.text())
      .then((body) => live && setText(body))
      .catch(() => undefined)
    return () => {
      live = false
    }
  }, [url])
  return <pre className="md">{text}</pre>
}
