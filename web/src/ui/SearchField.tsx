import type { Options, SearchSettings } from '../api'
import { Field } from './Field'
import { ModelFacts } from './ModelFacts'
import { choices, docFor, rerankerOptions } from './options'
import { Picker } from './Picker'
import { SEARCH_BOUNDS } from './searchFields'
import { Toggle } from './Toggle'

/** A number setting: the design's `.field` with a number input in it. */
export function Num({
  label,
  help,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  help?: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <Field label={label} help={help}>
      <input className="input" type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
    </Field>
  )
}

/** One search setting, as the settings page and the first run both show it. */
export function SearchField({
  name,
  search,
  options,
  onChange,
}: {
  name: keyof SearchSettings
  search: SearchSettings
  options: Options
  onChange: (next: SearchSettings) => void
}) {
  const doc = docFor(options.docs, `search.${name}`)
  switch (name) {
    case 'mode':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.search_modes)} value={search.mode} onChange={(mode) => onChange({ ...search, mode })} />
        </Field>
      )
    case 'fusion':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.fusions)} value={search.fusion} onChange={(fusion) => onChange({ ...search, fusion })} />
        </Field>
      )
    case 'rerank_excerpts':
    case 'rerank_with_context':
      return (
        <div className="field">
          <Toggle label={doc.title} checked={search[name]} onChange={(checked) => onChange({ ...search, [name]: checked })} />
          <span className="faint">{doc.description}</span>
        </div>
      )
    case 'min_rerank_score':
      // empty is a value: the chosen reranker's own calibrated floor
      return (
        <Field label={doc.title} help={doc.description}>
          <input
            className="input"
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={search.min_rerank_score ?? ''}
            placeholder="the reranker's floor"
            onChange={(event) => onChange({ ...search, min_rerank_score: event.target.value === '' ? null : Number(event.target.value) })}
          />
        </Field>
      )
    case 'fill_values':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.fill_values)} value={search.fill_values} onChange={(fill_values) => onChange({ ...search, fill_values })} />
        </Field>
      )
    case 'score_fold':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.score_folds)} value={search.score_fold} onChange={(score_fold) => onChange({ ...search, score_fold })} />
        </Field>
      )
    case 'reranker':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.rerankers)} value={search.reranker} onChange={(reranker) => onChange({ ...search, reranker })} />
        </Field>
      )
    case 'reranker_model':
    case 'map_reranker_model':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker
            ariaLabel={doc.title}
            options={rerankerOptions(options.reranker_models, options.reranker_metadata)}
            value={search[name]}
            onChange={(model) => onChange({ ...search, [name]: model })}
          />
          <ModelFacts name={search[name]} metadata={options.reranker_metadata[search[name]]} />
        </Field>
      )
    default: {
      const bounds = SEARCH_BOUNDS[name]
      return (
        <Num
          label={doc.title}
          help={doc.description}
          value={search[name]}
          min={bounds.min}
          max={bounds.max}
          step={bounds.step}
          onChange={(value) => onChange({ ...search, [name]: value })}
        />
      )
    }
  }
}
