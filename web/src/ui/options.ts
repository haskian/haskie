import type { EmbedderMetadata, EmbeddingModel, EmbeddingProfile, FieldDoc, ModelMetadata, RerankerMetadata } from '../api'
import { count } from '../format'
import type { PickerOption } from './Picker'

// Picker options and field texts, built from what `/api/options` offers.

// The backend names every setting; a key it does not know falls back to the key itself.
export const docFor = (docs: Record<string, FieldDoc>, key: string): FieldDoc => docs[key] ?? { title: key, description: '' }

/** Plain values as picker options, each labelled by itself. */
export const choices = <T extends string>(values: readonly T[]): PickerOption<T>[] => values.map((value) => ({ value, label: value }))

/** A model as a picker names it: its name without the publisher, an embedder's vector size,
 *  and its parameter count. */
export function modelLabel(name: string, metadata: ModelMetadata | undefined, dims?: number): string {
  return [name.split('/').at(-1), dims !== undefined && `Dim ${dims}`, metadata && `Param ${count(metadata.parameters)}`]
    .filter(Boolean)
    .join(' · ')
}

/** An embedding profile as a picker names it: its model, or full-text only for none. The profile
 *  key itself is only what the settings store. */
export const profileLabel = (model: EmbeddingModel | null | undefined, metadata: EmbedderMetadata | undefined): string =>
  model ? modelLabel(model.name, metadata, model.dims) : 'full-text only'

/** A reranker model as a picker option: labelled by `modelLabel`, described by its metadata. */
export const rerankerOption = (name: string, metadata: Record<string, RerankerMetadata>): PickerOption<string> => ({
  value: name,
  label: modelLabel(name, metadata[name]),
  sub: metadata[name]?.description,
})

export const rerankerOptions = (names: readonly string[], metadata: Record<string, RerankerMetadata>): PickerOption<string>[] =>
  names.map((name) => rerankerOption(name, metadata))

/** The embedding profiles as picker options, in the order the backend gives them, each described
 *  by its metadata (`Options.embedding_metadata`). */
export const profileOptions = (
  profiles: Record<string, EmbeddingModel | null>,
  metadata: Record<string, EmbedderMetadata>,
): PickerOption<EmbeddingProfile>[] =>
  Object.entries(profiles).map(([value, model]) => ({
    value,
    label: profileLabel(model, metadata[value]),
    sub: model ? metadata[value]?.description : 'Full-text search only. No model download.',
  }))
