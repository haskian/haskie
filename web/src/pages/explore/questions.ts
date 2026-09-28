/** The most questions one excerpts search takes, as the backend caps them (`MAX_QUESTIONS`). */
export const MAX_ASPECTS = 5

/** The questions an excerpts search asks: one an input, blank ones and repeats left out, as the
 *  backend would leave them out. */
export function questionsOf(aspects: string[]): string[] {
  return [...new Set(aspects.map((line) => line.trim()).filter(Boolean))]
}
