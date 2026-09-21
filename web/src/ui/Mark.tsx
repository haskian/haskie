import { markTerms } from './markTerms'

/** The text of a result, with the query's terms highlighted. */
export function Mark({ text, query }: { text: string; query: string }) {
  return <>{markTerms(text, query)}</>
}
