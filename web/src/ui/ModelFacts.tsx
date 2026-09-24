import type { EmbeddingModel, ModelCard } from '../api'
import { Kv } from './Kv'

/** One model's facts under its picker: its name, an embedder's vector size, then the card's
 *  key-value metadata in the order the backend gives it (`settings.ModelCard`). */
export function ModelFacts({ name, card, dims }: { name: string; card: ModelCard | null | undefined; dims?: number }) {
  const rows: [string, string][] = [['Model', name]]
  if (dims !== undefined) rows.push(['Dimensions', `${dims}`])
  rows.push(...Object.entries(card?.metadata ?? {}))
  return (
    <div className="model-card">
      <Kv rows={rows} />
    </div>
  )
}

/** An embedder's facts, or nothing for the full-text-only profile. */
export function EmbedderFacts({ model }: { model: EmbeddingModel | null | undefined }) {
  return model ? <ModelFacts name={model.name} card={model.card} dims={model.dims} /> : null
}
