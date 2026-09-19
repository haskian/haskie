import { useEffect, useState } from 'react'
import {
  api,
  type Accelerator,
  type Options,
  type PipelineSettings,
  type RetentionSettings,
  type UserSettings,
} from '../api'
import { ChunkSettingsForm } from '../components/ChunkSettingsForm'
import { Field } from '../components/Field'
import { ImportDefaultsForm } from '../components/ImportDefaultsForm'
import { SearchSettingsForm } from '../components/SearchSettingsForm'

export function Settings() {
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    api.settings().then(setSettings)
    api.options().then(setOptions)
  }, [])

  if (!settings || !options) return null
  const s = settings
  const docs = options.docs
  const update = (patch: Partial<UserSettings>) => {
    setSaved(false)
    setSettings({ ...s, ...patch })
  }
  const pipeline = (patch: Partial<PipelineSettings>) => update({ pipeline: { ...s.pipeline, ...patch } })
  const retention = (patch: Partial<RetentionSettings>) => update({ retention: { ...s.retention, ...patch } })

  return (
    <form
      className="settings"
      onSubmit={async (e) => {
        e.preventDefault()
        setSettings(await api.saveSettings(s))
        setSaved(true)
      }}
    >
      <h2>User settings</h2>

      <h3>Embedding</h3>
      <Field name="embedding" doc={docs['embedding']}>
        <select value={s.embedding} onChange={(e) => update({ embedding: e.target.value as UserSettings['embedding'] })}>
          {Object.entries(options.embedding_profiles).map(([p, m]) => (
            <option key={p} value={p}>
              {p} {m ? `(${m.name}, ${m.dims} dims)` : ''}
            </option>
          ))}
        </select>
      </Field>
      <Field name="accelerator" doc={docs['pipeline.accelerator']}>
        <select value={s.pipeline.accelerator} onChange={(e) => pipeline({ accelerator: e.target.value as Accelerator })}>
          {options.accelerators.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </Field>

      <h3>Import defaults (parser, skip OCR)</h3>
      <ImportDefaultsForm value={s.conversion} options={options} onChange={(next) => update({ conversion: next })} />

      <h3>Chunking defaults (per-collection overridable)</h3>
      <ChunkSettingsForm value={s.conversion} options={options} onChange={(next) => update({ conversion: next })} />

      <h3>Search (per-collection overridable)</h3>
      <SearchSettingsForm value={s.search} options={options} onChange={(next) => update({ search: next as UserSettings['search'] })} />

      <h3>Indexing parallelism</h3>
      <Field name="cpu_budget" doc={docs['pipeline.cpu_budget']}>
        <input type="number" min={1} value={s.pipeline.cpu_budget} onChange={(e) => pipeline({ cpu_budget: Number(e.target.value) })} />
      </Field>
      <Field name="converting_weight" doc={docs['pipeline.converting_weight']}>
        <input type="number" min={1} value={s.pipeline.converting_weight} onChange={(e) => pipeline({ converting_weight: Number(e.target.value) })} />
      </Field>
      <Field name="embedding_weight" doc={docs['pipeline.embedding_weight']}>
        <input type="number" min={1} value={s.pipeline.embedding_weight} onChange={(e) => pipeline({ embedding_weight: Number(e.target.value) })} />
      </Field>
      <Field name="indexing_weight" doc={docs['pipeline.indexing_weight']}>
        <input type="number" min={1} value={s.pipeline.indexing_weight} onChange={(e) => pipeline({ indexing_weight: Number(e.target.value) })} />
      </Field>
      <Field name="document_parallelism" doc={docs['pipeline.document_parallelism']}>
        <input type="number" min={0} value={s.pipeline.document_parallelism} onChange={(e) => pipeline({ document_parallelism: Number(e.target.value) })} />
      </Field>
      <Field name="batch_pages" doc={docs['pipeline.batch_pages']}>
        <input type="number" min={1} value={s.pipeline.batch_pages} onChange={(e) => pipeline({ batch_pages: Number(e.target.value) })} />
      </Field>
      <Field name="index_group_parts" doc={docs['pipeline.index_group_parts']}>
        <input type="number" min={1} value={s.pipeline.index_group_parts} onChange={(e) => pipeline({ index_group_parts: Number(e.target.value) })} />
      </Field>
      <Field name="task_timeout_seconds" doc={docs['pipeline.task_timeout_seconds']}>
        <input type="number" min={1} value={s.pipeline.task_timeout_seconds} onChange={(e) => pipeline({ task_timeout_seconds: Number(e.target.value) })} />
      </Field>
      <Field name="preview_workers" doc={docs['pipeline.preview_workers']}>
        <input type="number" min={1} value={s.pipeline.preview_workers} onChange={(e) => pipeline({ preview_workers: Number(e.target.value) })} />
      </Field>

      <h3>Index maintenance</h3>
      <Field name="maintenance_docs" doc={docs['pipeline.maintenance_docs']}>
        <input type="number" min={1} value={s.pipeline.maintenance_docs} onChange={(e) => pipeline({ maintenance_docs: Number(e.target.value) })} />
      </Field>
      <Field name="maintenance_idle_seconds" doc={docs['pipeline.maintenance_idle_seconds']}>
        <input type="number" min={1} value={s.pipeline.maintenance_idle_seconds} onChange={(e) => pipeline({ maintenance_idle_seconds: Number(e.target.value) })} />
      </Field>
      <Field name="ann_min_rows" doc={docs['pipeline.ann_min_rows']}>
        <input type="number" min={1} value={s.pipeline.ann_min_rows} onChange={(e) => pipeline({ ann_min_rows: Number(e.target.value) })} />
      </Field>

      <h3>Retention</h3>
      <Field name="job_days" doc={docs['retention.job_days']}>
        <input type="number" min={1} value={s.retention.job_days} onChange={(e) => retention({ job_days: Number(e.target.value) })} />
      </Field>
      <Field name="job_live_hours" doc={docs['retention.job_live_hours']}>
        <input type="number" min={2} value={s.retention.job_live_hours} onChange={(e) => retention({ job_live_hours: Number(e.target.value) })} />
      </Field>
      <Field name="audit_days" doc={docs['retention.audit_days']}>
        <input type="number" min={0} value={s.retention.audit_days} onChange={(e) => retention({ audit_days: Number(e.target.value) })} />
      </Field>

      <button>Save</button> {saved && <span className="muted">saved</span>}
    </form>
  )
}
