import type { Chunker, ChunkOverrides, ChunkSettings, Options } from '../api'
import { Field, NumberField } from './Field'

// Either the user's concrete defaults, or one collection's overrides of them: given `defaults`,
// a field may be left empty and falls back to them — which is what its placeholder shows.
type Props<V extends ChunkOverrides> = {
  value: V
  defaults?: ChunkSettings
  options: Options
  onChange: (next: V) => void
}

// How a document is split into chunks. These three values key the embedding cache, so the form
// is the same wherever they are set; only their emptiness means something different.
export function ChunkSettingsForm<V extends ChunkOverrides>({ value, defaults, options, onChange }: Props<V>) {
  const nullable = defaults !== undefined
  const doc = (k: keyof ChunkSettings) => options.docs[`conversion.${k}`]
  const set = <K extends keyof ChunkSettings>(k: K, v: ChunkSettings[K] | null) => onChange({ ...value, [k]: v })
  const placeholder = (k: keyof ChunkSettings) => (defaults ? `default ${String(defaults[k])}` : undefined)
  const num = (k: 'chunk_size' | 'chunk_overlap', min: number) => (
    <NumberField
      key={k}
      doc={doc(k)}
      value={value[k]}
      min={min}
      placeholder={placeholder(k)}
      onChange={(next) => set(k, next ?? (nullable ? null : 0))}
    />
  )
  return (
    <>
      <Field doc={doc('chunker')}>
        <select value={value.chunker ?? ''} onChange={(e) => set('chunker', (e.target.value || null) as Chunker | null)}>
          {nullable && <option value="">{placeholder('chunker')}</option>}
          {options.chunkers.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </Field>
      {num('chunk_size', 1)}
      {num('chunk_overlap', 0)}
    </>
  )
}
