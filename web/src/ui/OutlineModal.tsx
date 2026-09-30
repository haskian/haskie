import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { api, type Document, type OutlineSection } from '../api'
import { errorText } from '../format'
import type { Anchor } from './anchor'
import { DocumentPanes } from './DocumentPanes'
import { cite, lastHeading } from './match'
import { Modal } from './Modal'
import type { OpenedSection } from './SectionGrid'
import { outlineIndex } from './sections'
import { Skeleton } from './Skeleton'
import { Tabs, type TabDef } from './Tabs'

const OUTLINE_TAB = 'modal-outline'
const DOCUMENT_TAB = 'modal-document'
const TABS: TabDef[] = [
  { id: OUTLINE_TAB, label: 'Outline' },
  { id: DOCUMENT_TAB, label: 'Document' },
]

// `--depth` indents a row under its parent.
const depthStyle = (depth: number): CSSProperties => ({ '--depth': depth }) as CSSProperties

/** A section of the map, opened: its document's whole outline, every section with the words that
 *  say what it is about, the opened one marked and scrolled to. A row opens the document there. */
export function OutlineModal({ section, onClose }: { section: OpenedSection | null; onClose: () => void }) {
  return (
    <Modal open={section !== null} onClose={onClose} title={section?.document ?? ''} subtitle="outline">
      {section !== null && <OutlineBody key={`${section.document}:${section.line_start}`} section={section} />}
    </Modal>
  )
}

function OutlineBody({ section }: { section: OpenedSection }) {
  const [tab, setTab] = useState(OUTLINE_TAB)
  const [outline, setOutline] = useState<OutlineSection[] | null>(null)
  const [row, setRow] = useState<Document | null>(null) // the panes need the preview kind
  const [error, setError] = useState<string | null>(null)
  const [anchor, setAnchor] = useState<Anchor>({ heading: lastHeading(section.header) })
  const marked = useRef<HTMLDivElement>(null)
  const name = section.document

  useEffect(() => {
    let live = true
    Promise.all([api.outline(name), api.document(name)])
      .then(([sections, fetched]) => {
        if (!live) return
        setOutline(sections)
        setRow(fetched)
      })
      .catch((cause: unknown) => live && setError(errorText(cause)))
    return () => {
      live = false
    }
  }, [name])

  // The panel scrolls, not the modal: `scrollIntoView` would scroll the title out of sight too.
  useEffect(() => {
    const row = marked.current
    const panel = row?.closest('[role="tabpanel"]')
    if (!row || !panel) return
    const [at, within] = [row.getBoundingClientRect(), panel.getBoundingClientRect()]
    panel.scrollTop += at.top - within.top - (within.height - at.height) / 2
  }, [outline])

  const open = (entry: OutlineSection) => {
    setAnchor({ heading: lastHeading(entry.header) })
    setTab(DOCUMENT_TAB)
  }

  const at = outline === null ? -1 : outlineIndex(outline, section)
  return (
    <>
      <Tabs tabs={TABS} selected={tab} onSelect={setTab} />
      <div id={OUTLINE_TAB} role="tabpanel" className="match" hidden={tab !== OUTLINE_TAB}>
        {error !== null && <p className="muted">{error}</p>}
        {outline === null && error === null && <Skeleton />}
        {outline !== null && outline.length === 0 && <p className="muted">No outline yet: the document is still importing.</p>}
        {outline !== null && outline.length > 0 && (
          <div className="sections">
            {outline.map((entry, index) => (
              <div
                key={index}
                ref={index === at ? marked : undefined}
                className="outline-row"
                style={depthStyle(entry.depth)}
                role="button"
                tabIndex={0}
                aria-current={index === at}
                onClick={() => open(entry)}
              >
                <div className="outline-entry">
                  <span className="section-title">{entry.depth === 0 ? name : lastHeading(entry.header)}</span>
                  {entry.keywords.length > 0 && (
                    <div className="keywords">
                      {entry.keywords.map((word) => (
                        <span key={word} className="keyword">
                          {word}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <span className="mono muted">{cite(entry.location, name)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        {row !== null && <DocumentPanes doc={row.name} preview={row.preview} full anchor={anchor} shown={tab === DOCUMENT_TAB} />}
      </div>
    </>
  )
}
