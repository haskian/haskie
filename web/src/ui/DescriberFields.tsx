import type { Options, PipelineSettings } from '../api'
import { Field } from './Field'
import { choices, docFor } from './options'
import { Picker } from './Picker'

export type Describing = Pick<PipelineSettings, 'descriptors' | 'describer'>

/** Who describes the sections, as the settings page and the first run both show it: the strategy,
 *  and the describer model when the strategy is a language model. */
export function DescriberFields({
  value,
  options,
  onChange,
}: {
  value: Describing
  options: Options
  onChange: (next: Describing) => void
}) {
  const strategy = docFor(options.docs, 'pipeline.descriptors')
  const model = docFor(options.docs, 'pipeline.describer')
  return (
    <>
      <Field label={strategy.title} help={strategy.description}>
        <Picker
          ariaLabel={strategy.title}
          options={choices(options.descriptors)}
          value={value.descriptors}
          onChange={(descriptors) => onChange({ ...value, descriptors })}
        />
      </Field>
      {value.descriptors === 'llm' && (
        <Field label={model.title} help={model.description}>
          <Picker
            ariaLabel={model.title}
            options={choices(options.describers)}
            value={value.describer}
            onChange={(describer) => onChange({ ...value, describer })}
          />
        </Field>
      )}
    </>
  )
}
