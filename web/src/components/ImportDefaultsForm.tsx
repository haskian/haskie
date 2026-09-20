import type { ConversionSettings, Options, Parser } from '../api'
import { Field } from './Field'

type Props = {
  value: ConversionSettings
  options: Options
  onChange: (next: ConversionSettings) => void
}

// How a document is converted when nothing is said at import. Conversion runs once per document,
// so only the user defaults set these: a collection cannot change them afterwards.
export function ImportDefaultsForm({ value, options, onChange }: Props) {
  const doc = (k: keyof ConversionSettings) => options.docs[`conversion.${k}`]
  const set = <K extends keyof ConversionSettings>(k: K, v: ConversionSettings[K]) => onChange({ ...value, [k]: v })
  return (
    <>
      <Field doc={doc('parser')}>
        <select value={value.parser} onChange={(e) => set('parser', e.target.value as Parser)}>
          {options.parsers.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </Field>
      <Field doc={doc('skip_ocr_pages')}>
        <input type="checkbox" checked={value.skip_ocr_pages} onChange={(e) => set('skip_ocr_pages', e.target.checked)} />
      </Field>
    </>
  )
}
