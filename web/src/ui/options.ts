import type { EmbeddingModel, EmbeddingProfile, FieldDoc, ModelCard } from '../api'
import type { PickerOption } from './Picker'

// Picker options and field texts, built from what `/api/options` offers.

// The backend names every setting; a key it does not know falls back to the key itself.
export const docFor = (docs: Record<string, FieldDoc>, key: string): FieldDoc => docs[key] ?? { title: key, description: '' }

/** Plain values as picker options, each labelled by itself. */
export const choices = <T extends string>(values: readonly T[]): PickerOption<T>[] => values.map((value) => ({ value, label: value }))

/** A model as a picker names it: its name without the publisher, an embedder's vector size,
 *  and the parameter count its card gives. */
export function modelLabel(name: string, card: ModelCard | null | undefined, dims?: number): string {
  const params = card?.metadata.Parameters
  return [name.split('/').at(-1), dims !== undefined && `Dim ${dims}`, params && `Param ${params}`].filter(Boolean).join(' · ')
}

/** An embedding profile as a picker names it: its model, or full-text only for none. The profile
 *  key itself is only what the settings store. */
export const profileLabel = (model: EmbeddingModel | null | undefined): string =>
  model ? modelLabel(model.name, model.card, model.dims) : 'full-text only'

/** A reranker model as a picker option: labelled by `modelLabel`, described by its card. */
export const rerankerOption = (name: string, cards: Record<string, ModelCard>): PickerOption<string> => ({
  value: name,
  label: modelLabel(name, cards[name]),
  sub: cards[name]?.description,
})

export const rerankerOptions = (names: readonly string[], cards: Record<string, ModelCard>): PickerOption<string>[] =>
  names.map((name) => rerankerOption(name, cards))

/** The embedding profiles as picker options, in the order the backend gives them. */
export const profileOptions = (profiles: Record<string, EmbeddingModel | null>): PickerOption<EmbeddingProfile>[] =>
  Object.entries(profiles).map(([value, model]) => ({
    value: value as EmbeddingProfile,
    label: profileLabel(model),
    sub: model?.card?.description ?? 'Full-text search only. No model download.',
  }))
