import { useState } from 'react'
import { DocumentByName } from './DocumentPanes'
import { cite, lastHeading, WHOLE_DOCUMENT } from './match'
import { Modal } from './Modal'
import { Descriptors, OpenedDetail, type OpenedSection } from './SectionGrid'
import { Tabs, type TabDef } from './Tabs'

const SECTION_TAB = 'modal-section'
const DOCUMENT_TAB = 'modal-document'
const TABS: TabDef[] = [
  { id: SECTION_TAB, label: 'Section' },
  { id: DOCUMENT_TAB, label: 'Document' },
]

/** A section of the map, opened: what the map said about it (what it is about, its score, its
 *  matched chunks, the sections it covers), and its document open at its heading. A related
 *  section opens in its place. */
export function SectionsModal({
  section,
  onClose,
  onOpen,
}: {
  section: OpenedSection | null
  onClose: () => void
  onOpen?: (section: OpenedSection) => void
}) {
  return (
    <Modal open={section !== null} onClose={onClose} title={section?.document ?? ''} subtitle="section">
      {section !== null && <SectionBody key={`${section.collection}:${section.id}`} section={section} onOpen={onOpen} />}
    </Modal>
  )
}

function SectionBody({ section, onOpen }: { section: OpenedSection; onOpen?: (section: OpenedSection) => void }) {
  const [tab, setTab] = useState(SECTION_TAB)
  const name = section.document

  return (
    <>
      <Tabs tabs={TABS} selected={tab} onSelect={setTab} />
      <div id={SECTION_TAB} role="tabpanel" className="match" hidden={tab !== SECTION_TAB}>
        <div className="opened-detail">
          <p className="section-heading">{section.header || WHOLE_DOCUMENT}</p>
          <span className="mono muted">{cite(section.location, name)}</span>
          {'descriptors' in section && <Descriptors words={section.descriptors} />}
          <OpenedDetail section={section} onOpen={onOpen} />
        </div>
      </div>
      <div id={DOCUMENT_TAB} role="tabpanel" hidden={tab !== DOCUMENT_TAB}>
        <DocumentByName name={name} anchor={{ heading: lastHeading(section.header) }} shown={tab === DOCUMENT_TAB} />
      </div>
    </>
  )
}
