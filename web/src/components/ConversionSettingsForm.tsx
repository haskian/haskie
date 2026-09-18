import type { Chunker, ConversionOverrides, ConversionSettings, Options, Parser } from '../api'
import { Field } from './Field'

type Props = {
  value: ConversionSettings | ConversionOverrides
  defaults?: ConversionSettings // when set, fields are overrides and may be null (= default)
  options: Options
  onChange: (next: ConversionSettings | ConversionOverrides) => void
}

// One form for both user-level (concrete values) and library-level (nullable overrides) settings.
export function ConversionSettingsForm({ value, defaults, options, onChange }: Props) {
  const doc = (k: keyof ConversionSettings) => options.docs[`defaults.${k}`]
  const set = <K extends keyof ConversionSettings>(k: K, v: ConversionSettings[K] | null) => onChange({ ...value, [k]: v })
  const placeholder = (k: keyof ConversionSettings) => (defaults ? `default ${String(defaults[k])}` : undefined)
  const choice = <K extends 'parser' | 'chunker'>(k: K, items: readonly string[]) => (
    <Field key={k} name={k} doc={doc(k)}>
      <select value={value[k] ?? ''} onChange={(e) => set(k, (e.target.value || null) as ConversionSettings[K] | null)}>
        {defaults && <option value="">{placeholder(k)}</option>}
        {items.map((m) => <option key={m} value={m}>{m}</option>)}
      </select>
    </Field>
  )
  const num = (k: 'chunk_size' | 'chunk_overlap', min: number) => (
    <Field key={k} name={k} doc={doc(k)}>
      <input
        type="number"
        min={min}
        value={value[k] ?? ''}
        placeholder={placeholder(k)}
        onChange={(e) => set(k, e.target.value === '' ? (defaults ? null : 0) : Number(e.target.value))}
      />
    </Field>
  )
  return (
    <>
      {choice('parser', options.parsers as Parser[])}
      {choice('chunker', options.chunkers as Chunker[])}
      {num('chunk_size', 1)}
      {num('chunk_overlap', 0)}
      <Field name="skip_ocr_pages" doc={doc('skip_ocr_pages')}>
        {defaults ? (
          <select
            value={value.skip_ocr_pages === null ? '' : String(value.skip_ocr_pages)}
            onChange={(e) => set('skip_ocr_pages', e.target.value === '' ? null : e.target.value === 'true')}
          >
            <option value="">default ({defaults.skip_ocr_pages ? 'yes' : 'no'})</option>
            <option value="true">yes</option>
            <option value="false">no</option>
          </select>
        ) : (
          <input type="checkbox" checked={!!value.skip_ocr_pages} onChange={(e) => set('skip_ocr_pages', e.target.checked)} />
        )}
      </Field>
    </>
  )
}
