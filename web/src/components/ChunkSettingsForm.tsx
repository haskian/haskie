import type { Chunker, ChunkOverrides, ChunkSettings, Options } from '../api'
import { Field } from './Field'

// Either the user's concrete defaults, or one collection's overrides of them: `nullable` fields
// may be left empty, and fall back to `defaults` — which is what their placeholder shows.
type Props<V extends ChunkOverrides> = {
  value: V
  options: Options
  onChange: (next: V) => void
} & ({ nullable: true; defaults: ChunkSettings } | { nullable?: false; defaults?: never })

// How a document is split into chunks. These three values key the embedding cache, so the form
// is the same wherever they are set; only their emptiness means something different.
export function ChunkSettingsForm<V extends ChunkOverrides>({ value, options, onChange, nullable, defaults }: Props<V>) {
  const doc = (k: keyof ChunkSettings) => options.docs[`conversion.${k}`]
  const set = <K extends keyof ChunkSettings>(k: K, v: ChunkSettings[K] | null) => onChange({ ...value, [k]: v })
  const placeholder = (k: keyof ChunkSettings) => (defaults ? `default ${String(defaults[k])}` : undefined)
  const num = (k: 'chunk_size' | 'chunk_overlap', min: number) => (
    <Field key={k} name={k} doc={doc(k)}>
      <input
        type="number"
        min={min}
        value={value[k] ?? ''}
        placeholder={placeholder(k)}
        onChange={(e) => set(k, e.target.value === '' ? (nullable ? null : 0) : Number(e.target.value))}
      />
    </Field>
  )
  return (
    <>
      <Field name="chunker" doc={doc('chunker')}>
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
