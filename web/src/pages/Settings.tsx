import { useEffect, useState } from 'react'
import { api, type Accelerator, type PipelineSettings, type RetentionSettings, type UserSettings } from '../api'
import { ChunkSettingsForm } from '../components/ChunkSettingsForm'
import { Field, NumberField } from '../components/Field'
import { ImportDefaultsForm } from '../components/ImportDefaultsForm'
import { SearchSettingsForm } from '../components/SearchSettingsForm'
import { useOptions } from '../hooks/useOptions'
import { useRun } from '../hooks/useRun'

// Setting key and its minimum, in the order they are shown. The definitions come from
// /api/options.docs; only the bounds live here, and they are the backend's (see settings.py).
type PipelineNumber = Exclude<keyof PipelineSettings, 'accelerator'>
const PARALLELISM: readonly (readonly [PipelineNumber, number])[] = [
  ['cpu_budget', 1],
  ['converting_weight', 1],
  ['embedding_weight', 1],
  ['indexing_weight', 1],
  ['document_parallelism', 0],
  ['batch_pages', 1],
  ['index_group_parts', 1],
  ['task_timeout_seconds', 1],
  ['preview_workers', 1],
]
const MAINTENANCE: readonly (readonly [PipelineNumber, number])[] = [
  ['maintenance_docs', 1],
  ['maintenance_idle_seconds', 1],
  ['ann_min_rows', 1],
]
const RETENTION: readonly (readonly [keyof RetentionSettings, number])[] = [
  ['job_days', 1],
  ['audit_days', 0],
]

export function Settings() {
  const options = useOptions()
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [saved, setSaved] = useState(false)
  const { run, busy, error } = useRun(async () => setSaved(true))

  useEffect(() => {
    api.settings().then(setSettings)
  }, [])

  if (!settings) return null
  const s = settings
  const docs = options.docs
  const update = (patch: Partial<UserSettings>) => {
    setSaved(false)
    setSettings({ ...s, ...patch })
  }
  const pipeline = <K extends keyof PipelineSettings>(k: K, v: PipelineSettings[K]) =>
    update({ pipeline: { ...s.pipeline, [k]: v } })
  const retention = <K extends keyof RetentionSettings>(k: K, v: RetentionSettings[K]) =>
    update({ retention: { ...s.retention, [k]: v } })
  const numbers = (table: readonly (readonly [PipelineNumber, number])[]) =>
    table.map(([k, min]) => (
      <NumberField key={k} doc={docs[`pipeline.${k}`]} value={s.pipeline[k]} min={min} onChange={(v) => pipeline(k, v ?? 0)} />
    ))

  return (
    <form
      className="settings"
      onSubmit={(e) => {
        e.preventDefault()
        run(() => api.saveSettings(s).then(setSettings))
      }}
    >
      <h2>User settings</h2>
      {error && <p className="error">{error}</p>}

      <h3>Embedding</h3>
      <Field doc={docs['embedding']}>
        <select value={s.embedding} onChange={(e) => update({ embedding: e.target.value as UserSettings['embedding'] })}>
          {Object.entries(options.embedding_profiles).map(([p, m]) => (
            <option key={p} value={p}>
              {p} {m ? `(${m.name}, ${m.dims} dims)` : ''}
            </option>
          ))}
        </select>
      </Field>
      <Field doc={docs['pipeline.accelerator']}>
        <select value={s.pipeline.accelerator} onChange={(e) => pipeline('accelerator', e.target.value as Accelerator)}>
          {options.accelerators.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </Field>

      <h3>Import defaults (parser, skip OCR)</h3>
      <ImportDefaultsForm value={s.conversion} options={options} onChange={(next) => update({ conversion: next })} />

      <h3>Chunking defaults (per-collection overridable)</h3>
      <ChunkSettingsForm value={s.conversion} options={options} onChange={(next) => update({ conversion: next })} />

      <h3>Search (per-collection overridable)</h3>
      <SearchSettingsForm value={s.search} options={options} onChange={(next) => update({ search: next })} />

      <h3>Indexing parallelism</h3>
      {numbers(PARALLELISM)}

      <h3>Index maintenance</h3>
      {numbers(MAINTENANCE)}

      <h3>Retention</h3>
      {RETENTION.map(([k, min]) => (
        <NumberField key={k} doc={docs[`retention.${k}`]} value={s.retention[k]} min={min} onChange={(v) => retention(k, v ?? 0)} />
      ))}

      <button disabled={busy}>Save</button> {saved && <span className="muted">saved</span>}
    </form>
  )
}
