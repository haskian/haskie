import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { Mark } from './Mark'
import { termPattern } from './markTerms'

/** Wraps every match of the query's terms in the text under `root` in `<mark>`, as `Mark` does for
 *  plain text: over the rendered markdown, so a term inside a link or a list item is marked too
 *  and no tag or attribute is touched. */
export function markTextNodes(root: HTMLElement, query: string): void {
  const pattern = termPattern(query)
  if (pattern === null) return
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const nodes: Text[] = []
  while (walker.nextNode()) nodes.push(walker.currentNode as Text)
  for (const node of nodes) {
    const parts = (node.textContent ?? '').split(pattern)
    if (parts.length === 1) continue
    const marked = parts.map((part, index) => {
      if (index % 2 === 0) return document.createTextNode(part)
      const mark = document.createElement('mark')
      mark.textContent = part
      return mark
    })
    node.replaceWith(...marked)
  }
}

/**
 * A passage or excerpt's text rendered as markdown, its headings, lists and emphasis as the
 * document has them, with the query's terms marked. The server renders it (raw HTML stripped, as
 * for a document page); until it answers, or if it fails, the plain text shows instead.
 */
export function MarkdownQuote({ text, query }: { text: string; query: string }) {
  const [html, setHtml] = useState<string | null>(null)
  const body = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let live = true
    api
      .renderMarkdown(text)
      .then((rendered) => live && setHtml(rendered.html))
      .catch(() => live && setHtml(null))
    return () => {
      live = false
    }
  }, [text])

  useEffect(() => {
    if (html !== null && body.current !== null) markTextNodes(body.current, query)
  }, [html, query])

  if (html === null) {
    return (
      <div className="markdown">
        <p>
          <Mark text={text} query={query} />
        </p>
      </div>
    )
  }
  // rendered by the server with raw HTML stripped first, so a document cannot inject markup here
  return <div className="markdown" ref={body} dangerouslySetInnerHTML={{ __html: html }} />
}
