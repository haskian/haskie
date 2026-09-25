import type { Options, SearchSettings } from '../api'
import { Field } from './Field'
import { ModelFacts } from './ModelFacts'
import { choices, docFor, rerankerOptions } from './options'
import { Picker } from './Picker'
import { SEARCH_BOUNDS } from './searchFields'

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
    case 'reranker':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker ariaLabel={doc.title} options={choices(options.rerankers)} value={search.reranker} onChange={(reranker) => onChange({ ...search, reranker })} />
        </Field>
      )
    case 'reranker_model':
      return (
        <Field label={doc.title} help={doc.description}>
          <Picker
            ariaLabel={doc.title}
            options={rerankerOptions(options.reranker_models, options.reranker_metadata)}
            value={search.reranker_model}
            onChange={(reranker_model) => onChange({ ...search, reranker_model })}
          />
          <ModelFacts name={search.reranker_model} metadata={options.reranker_metadata[search.reranker_model]} />
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
          step={bounds.step}
          onChange={(value) => onChange({ ...search, [name]: value })}
        />
      )
    }
  }
}
