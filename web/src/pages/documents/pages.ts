import type { Document } from '../../api'

/** A PDF's pages and how its conversion read them: converted, OCR's among them, and not
 *  converted (left out, as `skip_ocr_pages` allows). Null for other formats. A PDF converted
 *  before the counts were kept shows its page count alone. */
export function pagesLabel(doc: Pick<Document, 'pages' | 'pages_ocr' | 'pages_unread'>): string | null {
  const { pages, pages_ocr: ocr, pages_unread: unread } = doc
  if (pages == null) return null
  if (ocr == null || unread == null) return `${pages}`
  return [
    `${pages - unread} of ${pages} converted`,
    ...(ocr > 0 ? [`${ocr} read by OCR`] : []),
    ...(unread > 0 ? [`${unread} not converted`] : []),
  ].join(' · ')
}
