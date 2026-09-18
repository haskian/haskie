import { useEffect, useState } from 'react'
import { api, type Head } from '../api'
import type { Route } from '../App'

// The server numbers heading ids by position (`render.to_html`), so the nth entry of the table of
// contents links to `h-n`. Nothing here counts renders: the position in the document is fixed.
const anchorId = (index: number) => `h-${index}`

export function Viewer({ library, doc, navigate }: { library: string; doc: string; navigate: (p: Route) => void }) {
  const [head, setHead] = useState<Head | null>(null)
  const [pages, setPages] = useState<string[]>([])
  const [full, setFull] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    const read = async () => {
      // cleared here rather than in the effect body: this is the start of the new request, and
      // the old document stays on screen until its replacement begins to arrive
      setHead(null)
      setPages([])
      setError(null)
      for await (const frame of api.markdown(library, doc, full)) {
        if (!live) return // the document changed while this one was still arriving
        if (frame.kind === 'head') setHead(frame)
        else setPages((shown) => [...shown, frame.html])
      }
    }
    read().catch((e: unknown) => live && setError(String(e)))
    return () => {
      live = false
    }
  }, [library, doc, full])

  const preview = head?.preview
  const previewUrl = api.previewUrl(library, doc)

  return (
    <div className="viewer">
      <header>
        <button onClick={() => navigate({ name: 'libraries', library })}>← {library}</button>
        <strong>{doc}</strong>
        {preview?.truncated && (
          <span className="banner">preview: first {preview.pages} pages — index the document for the full text</span>
        )}
        {preview && preview.ocr_pages.length > 0 && (
          <span className="banner">pages needing OCR: {preview.ocr_pages.join(', ')}</span>
        )}
        <label>
          <input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} /> full markdown (indexed only)
        </label>
        <a href={api.sourceUrl(library, doc)} target="_blank" rel="noreferrer">
          open original
        </a>
      </header>
      {error && <p className="error">{error}</p>}
      <div className="panes">
        <div className="pane">
          {preview?.kind === 'pdf' && <iframe title="source" src={previewUrl} />}
          {preview?.kind === 'html' && <iframe title="source" src={previewUrl} sandbox="" />}
          {preview?.kind === 'image' && <img alt={doc} src={previewUrl} />}
          {preview?.kind === 'text' && <TextPane url={previewUrl} />}
        </div>
        <div className="pane markdown">
          {head && head.toc.length > 0 && (
            <details open className="toc">
              <summary>Contents</summary>
              {head.toc.map((h, index) => (
                <div key={anchorId(index)} style={{ paddingLeft: `${(h.level - 1) * 12}px` }}>
                  <a href={`#${anchorId(index)}`}>{h.text}</a>
                </div>
              ))}
            </details>
          )}
          {/* The HTML is rendered by the server, which strips raw HTML first (see render.py), so
              there is nothing here for a document to inject. One element per page, appended as
              each frame arrives, so a long document shows its first page immediately. */}
          {pages.map((html, index) => (
            <div key={index} dangerouslySetInnerHTML={{ __html: html }} />
          ))}
          {head && pages.length < head.pages && (
            <p className="muted">
              {pages.length} of {head.pages} pages…
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function TextPane({ url }: { url: string }) {
  const [text, setText] = useState('')
  useEffect(() => {
    fetch(url).then((r) => r.text()).then(setText)
  }, [url])
  return <pre>{text}</pre>
}
