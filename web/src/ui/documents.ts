import { File, FileCode, FileText, Image as ImageIcon } from 'lucide-react'
import type { ComponentType } from 'react'

/** A lucide icon, as the components that take one accept it. */
export type IconComponent = ComponentType<{ className?: string }>

const IMAGE_SUFFIXES: readonly string[] = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff', '.svg']
const TEXT_SUFFIXES: readonly string[] = ['.md', '.txt']

/** The icon for a document, by the suffix of the file it was imported from. */
export function documentIcon(suffix: string): IconComponent {
  if (suffix === '.pdf') return FileText
  if (IMAGE_SUFFIXES.includes(suffix)) return ImageIcon
  if (TEXT_SUFFIXES.includes(suffix)) return FileCode
  return File
}

const LETTERS_PER_RANGE = 5
const LAST_LETTER_RANGE = 4 // U–Z holds six letters, so the last band absorbs Z
/** The gallery bands, in the order the pages show them. */
export const NAME_RANGES: readonly string[] = ['A–E', 'F–J', 'K–O', 'P–T', 'U–Z', '#']

/** The band a name belongs to: five letters each, and everything else under `#`. */
export function nameRange(name: string): string {
  const letter = name.trim().charAt(0).toUpperCase()
  if (letter < 'A' || letter > 'Z') return '#'
  return NAME_RANGES[Math.min(LAST_LETTER_RANGE, Math.floor((letter.charCodeAt(0) - 'A'.charCodeAt(0)) / LETTERS_PER_RANGE))]
}

/** One labelled band of a gallery. */
export interface RangeGroup<T> {
  label: string
  items: T[]
}

/** One band per name range that holds something; empty bands are left out. */
export function groupByRange<T>(items: T[], nameOf: (item: T) => string): RangeGroup<T>[] {
  return NAME_RANGES.map((label) => ({ label, items: items.filter((item) => nameRange(nameOf(item)) === label) })).filter(
    (group) => group.items.length > 0,
  )
}
