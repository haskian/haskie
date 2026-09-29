import type { ReactNode } from 'react'
import type { Device, EmbeddingModel, ModelMetadata, Runtime } from '../api'
import { Kv } from './Kv'
import { count } from '../format'

// what each needs beyond the default install, in its name, so the picker says it before a download
const RUNTIMES: Record<Runtime, string> = { onnx: 'ONNX', mlx: 'MLX (the mlx extra)', gguf: 'llama.cpp GGUF (the gguf extra)' }
const DEVICES: Record<Device, string> = { cpu: 'CPU', apple_silicon: 'Apple Silicon', gpu: 'GPU (CUDA)' }

/** One model's facts under its picker: its name, an embedder's vector size, then what the
 *  catalogue says about it (`catalogue.ModelMetadata`). */
export function ModelFacts({ name, metadata }: { name: string; metadata: ModelMetadata | undefined }) {
  const rows: [string, ReactNode][] = [['Model', name]]
  if (metadata) {
    if ('dimensions' in metadata) rows.push(['Dimensions', `${metadata.dimensions}`])
    rows.push(
      ['Parameters', count(metadata.parameters)],
      ['Context', `${count(metadata.context_tokens)} tokens`],
      ['Released', metadata.released],
      ['Languages', metadata.languages],
      ['License', metadata.license],
      ['Runtime', RUNTIMES[metadata.runtime]],
      ['Devices', metadata.devices.map((device) => DEVICES[device]).join(', ')],
      [
        'Model card',
        <a key="card" href={metadata.model_card_url} target="_blank" rel="noreferrer">
          {metadata.model_card_url.replace('https://', '')}
        </a>,
      ],
    )
  }
  return (
    <div className="model-card">
      <Kv rows={rows} />
    </div>
  )
}

/** An embedder's facts, or nothing for the full-text-only profile. */
export function EmbedderFacts({ model, metadata }: { model: EmbeddingModel | null | undefined; metadata: ModelMetadata | undefined }) {
  return model ? <ModelFacts name={model.name} metadata={metadata} /> : null
}
