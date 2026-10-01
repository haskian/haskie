import { useCallback, useEffect, useState } from 'react'
import {
  api,
  type Options,
  type PipelineSettings,
  type RetentionSettings,
  type UserSettings,
} from '../api'
import type { PageProps } from '../App'
import { errorText } from '../format'
import { choices, docFor, EmbedderFacts, Field, Num, Picker, profileOptions, SearchField, Shell, Toggle, visibleExpansionFields, visibleSearchFields, type NumericKeys } from '../ui'
import { classicBackground, setBackground as storeBackground } from './settings/background'
import './Settings.css'

const SECTIONS: { id: string; label: string }[] = [
  { id: 'embedding', label: 'Embedding' },
  { id: 'import', label: 'Import' },
  { id: 'chunking', label: 'Chunking' },
  { id: 'search', label: 'Search' },
  { id: 'expansion', label: 'Expansion' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'maintenance', label: 'Maintenance' },
  { id: 'retention', label: 'Retention' },
  { id: 'appearance', label: 'Appearance' },
]

interface NumberField<T> {
  key: NumericKeys<T>
  min: number
  step?: number
}

const PIPELINE_FIELDS: NumberField<PipelineSettings>[] = [
  { key: 'cpu_budget', min: 1 },
  { key: 'converting_weight', min: 1 },
  { key: 'embedding_weight', min: 1 },
  { key: 'indexing_weight', min: 1 },
  { key: 'document_parallelism', min: 0 },
  { key: 'batch_pages', min: 1 },
  { key: 'index_group_parts', min: 1 },
  { key: 'task_timeout_seconds', min: 1 },
  { key: 'preview_workers', min: 1 },
]

const MAINTENANCE_FIELDS: NumberField<PipelineSettings>[] = [
  { key: 'maintenance_documents', min: 1 },
  { key: 'maintenance_idle_seconds', min: 1 },
  { key: 'ann_min_rows', min: 1 },
]

const RETENTION_FIELDS: NumberField<RetentionSettings>[] = [
  { key: 'operation_days', min: 1 },
  { key: 'audit_days', min: 0 },
  { key: 'search_days', min: 0 },
]


/** The user settings, section by section. Every label and help text comes from `/api/options`. */
export function Settings({ route, counts, refreshStatus }: PageProps) {
  const [settings, setSettings] = useState<UserSettings | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [section, setSection] = useState(SECTIONS[0].id)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [classic, setClassic] = useState(classicBackground)

  const reload = useCallback(
    () =>
      api
        .settings()
        .then((loaded) => {
          setSettings(loaded)
          setSaved(false)
          setError(null)
        })
        .catch((failure: unknown) => setError(errorText(failure))),
    [],
  )

  useEffect(() => {
    void reload()
    api.options().then(setOptions).catch(() => undefined)
  }, [reload])

  const setBackground = (on: boolean): void => {
    setClassic(on)
    storeBackground(on)
  }

  // No scroll-spy. The nav marks what was last clicked, which is where the page went.
  const goTo = (id: string): void => {
    setSection(id)
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const side = (
    <section>
      <span className="label label-mono">Sections</span>
      <nav className="nav">
        {SECTIONS.map((entry) => (
          // A button, not an anchor: an `#embedding` href would be read as a route by the hash router.
          <button
            key={entry.id}
            className="nav-item"
            type="button"
            aria-current={entry.id === section ? 'true' : undefined}
            onClick={() => goTo(entry.id)}
          >
            {entry.label}
          </button>
        ))}
      </nav>
    </section>
  )

  if (settings === null || options === null) {
    return (
      <Shell current={route.name} counts={counts} side={side}>
        <div className="settings">{error !== null && <p className="muted">{error}</p>}</div>
      </Shell>
    )
  }

  const docs = options.docs
  const update = (patch: Partial<UserSettings>): void => {
    setSaved(false)
    setSettings({ ...settings, ...patch })
  }
  const pipeline = (patch: Partial<PipelineSettings>): void => update({ pipeline: { ...settings.pipeline, ...patch } })
  const retention = (patch: Partial<RetentionSettings>): void => update({ retention: { ...settings.retention, ...patch } })

  const save = (): void => {
    api
      .saveSettings(settings)
      .then((stored) => {
        setSettings(stored)
        setSaved(true)
        setError(null)
        void refreshStatus() // the profile or the reranker may need a model the status bar lacks
        api.options().then(setOptions).catch(() => undefined) // the models offered follow the hardware
      })
      .catch((failure: unknown) => setError(errorText(failure)))
  }

  const profiles = profileOptions(options.embedding_profiles, options.embedding_metadata)
  const skipOcr = docFor(docs, 'conversion.skip_ocr_pages')
  const frame = docFor(docs, 'conversion.chunk_frame')

  return (
    <Shell current={route.name} counts={counts} side={side}>
      <div className="settings">
        <section id="embedding">
          <span className="mono muted">Embedding</span>
          <Field label={docFor(docs, 'embedding').title} help={docFor(docs, 'embedding').description}>
            <Picker ariaLabel="Embedding profile" options={profiles} value={settings.embedding} onChange={(embedding) => update({ embedding })} />
            <EmbedderFacts model={options.embedding_profiles[settings.embedding]} metadata={options.embedding_metadata[settings.embedding]} />
          </Field>
          <Field label={docFor(docs, 'pipeline.accelerator').title} help={docFor(docs, 'pipeline.accelerator').description}>
            <Picker
              ariaLabel={docFor(docs, 'pipeline.accelerator').title}
              options={choices(options.accelerators)}
              value={settings.pipeline.accelerator}
              onChange={(accelerator) => pipeline({ accelerator })}
            />
          </Field>
        </section>

        <section id="import">
          <span className="mono muted">Import</span>
          <Field label={docFor(docs, 'conversion.parser').title} help={docFor(docs, 'conversion.parser').description}>
            <Picker
              ariaLabel="Parser"
              options={choices(options.parsers)}
              value={settings.conversion.parser}
              onChange={(parser) => update({ conversion: { ...settings.conversion, parser } })}
            />
          </Field>
          <div className="field">
            <Toggle
              label={skipOcr.title}
              checked={settings.conversion.skip_ocr_pages}
              onChange={(skip_ocr_pages) => update({ conversion: { ...settings.conversion, skip_ocr_pages } })}
            />
            <span className="faint">{skipOcr.description}</span>
          </div>
        </section>

        <section id="chunking">
          <span className="mono muted">Chunking</span>
          <Field label={docFor(docs, 'conversion.chunker').title} help={docFor(docs, 'conversion.chunker').description}>
            <Picker
              ariaLabel="Chunker"
              options={choices(options.chunkers)}
              value={settings.conversion.chunker}
              onChange={(chunker) => update({ conversion: { ...settings.conversion, chunker } })}
            />
          </Field>
          <Num
            label={docFor(docs, 'conversion.chunk_size').title}
            help={docFor(docs, 'conversion.chunk_size').description}
            value={settings.conversion.chunk_size}
            min={1}
            onChange={(chunk_size) => update({ conversion: { ...settings.conversion, chunk_size } })}
          />
          <Num
            label={docFor(docs, 'conversion.chunk_merge_below').title}
            help={docFor(docs, 'conversion.chunk_merge_below').description}
            value={settings.conversion.chunk_merge_below}
            min={0}
            max={100}
            onChange={(chunk_merge_below) => update({ conversion: { ...settings.conversion, chunk_merge_below } })}
          />
          <div className="field">
            <Toggle
              label={frame.title}
              checked={settings.conversion.chunk_frame}
              onChange={(chunk_frame) => update({ conversion: { ...settings.conversion, chunk_frame } })}
            />
            <span className="faint">{frame.description}</span>
          </div>
        </section>

        <section id="search">
          <span className="mono muted">Search</span>
          {visibleSearchFields(settings.search).map((name) => (
            <SearchField key={name} name={name} search={settings.search} options={options} onChange={(search) => update({ search })} />
          ))}
        </section>

        <section id="expansion">
          <span className="mono muted">Expansion</span>
          {visibleExpansionFields(settings.search).map((name) => (
            <SearchField key={name} name={name} search={settings.search} options={options} onChange={(search) => update({ search })} />
          ))}
        </section>

        <section id="pipeline">
          <span className="mono muted">Pipeline</span>
          {PIPELINE_FIELDS.map((field) => (
            <Num
              key={field.key}
              label={docFor(docs, `pipeline.${field.key}`).title}
              help={docFor(docs, `pipeline.${field.key}`).description}
              value={settings.pipeline[field.key]}
              min={field.min}
              onChange={(value) => pipeline({ [field.key]: value })}
            />
          ))}
        </section>

        <section id="maintenance">
          <span className="mono muted">Maintenance</span>
          {MAINTENANCE_FIELDS.map((field) => (
            <Num
              key={field.key}
              label={docFor(docs, `pipeline.${field.key}`).title}
              help={docFor(docs, `pipeline.${field.key}`).description}
              value={settings.pipeline[field.key]}
              min={field.min}
              onChange={(value) => pipeline({ [field.key]: value })}
            />
          ))}
        </section>

        <section id="retention">
          <span className="mono muted">Retention</span>
          {RETENTION_FIELDS.map((field) => (
            <Num
              key={field.key}
              label={docFor(docs, `retention.${field.key}`).title}
              help={docFor(docs, `retention.${field.key}`).description}
              value={settings.retention[field.key]}
              min={field.min}
              onChange={(value) => retention({ [field.key]: value })}
            />
          ))}
        </section>

        <section id="appearance">
          <span className="mono muted">Appearance</span>
          <div className="row row-loose">
            <Toggle id="bg-classic" label="Classic background" checked={classic} onChange={setBackground} />
          </div>
          <div className="row row-loose">
            <button className="btn btn-primary" type="button" onClick={save}>
              Save
            </button>
            <button className="btn btn-ghost" type="button" onClick={() => void reload()}>
              Reset
            </button>
            {saved && <span className="muted">saved</span>}
          </div>
          {error !== null && <p className="muted">{error}</p>}
        </section>
      </div>
    </Shell>
  )
}
