import { Mark } from './Mark'
import { alsoDocuments, alsoOf, everyPlace, fillOf, headingOf, isExcerpt, placeOf, questionLabels, scoreStyle, type Match } from './match'

// A result shows the text that matched. Under it, one line each: the heading it sits under, where
// to read it, and the chunks it is made of and how many other places say the same.
// How many other places say the same, and how many other documents they are in: "also in 4 / 2".
const alsoIn = (match: Match): string => {
  const places = everyPlace(alsoOf(match)).length
  return places > 0 ? ` · also in ${places} / ${alsoDocuments(match)}` : ''
}
const EMPTY = '\u00a0' // an empty line keeps its height, so every tile of a grid stays level
const linesOf = (match: Match): string[] => {
  const heading = headingOf(match) || match.header || EMPTY
  const [where, chunks] = placeOf(match)
  return [heading, where || EMPTY, `${chunks}${alsoIn(match)}`]
}
/** What the number on a question's tag means: the excerpt's best chunk for that question, on the
 *  reranker's scale when one scored it, else on that question's own search's scale. */
export function questionScoreMeaning(label: string, reranked: boolean): string {
  return reranked
    ? `The reranker's score for this excerpt's best chunk against ${label}: 0 to 1, higher is better. A question tags an excerpt only when the reranker scored it above its floor.`
    : `This excerpt's best chunk's score in ${label}'s own search: on that search's scale, so not comparable with another question's.`
}

// offsets are unique in a document
const keyOf = (match: Match): string => `${match.collection}:${match.document}:${match.char_start}`

/** The result grid, in any of its shapes: chunks, passages or excerpts that matched.
 *  With several `questions` asked, an excerpt names the ones it answers, a tag each with how well
 *  it matched that question ("Q1 0.84"), at the bottom of its tile. */
export function HitGrid<T extends Match>({
  results,
  query,
  onOpen,
  questions = [],
  reranked = false,
}: {
  results: T[]
  query: string
  onOpen?: (match: T) => void
  questions?: string[]
  reranked?: boolean // a reranker scored the results, so a question's score reads 0 to 1
}) {
  const scores = results.map((match) => match.score)
  const best = Math.max(...scores)
  const worst = Math.min(...scores)
  return (
    <div className="hits">
      {results.map((match) => (
        <article key={keyOf(match)} className="hit" style={scoreStyle(fillOf(match.score, best, worst))} onClick={() => onOpen?.(match)}>
          <header className="hit-head">
            <span className="tag">
              <span className="kind">{match.collection}</span>
              <span>{match.document}</span>
            </span>
            <span className="mono muted">{match.score.toFixed(2)}</span>
          </header>
          <div className="hit-body">
            <p className="hit-text">
              <Mark text={match.text} query={query} />
            </p>
            <footer className="hit-foot">
              {linesOf(match).map((line, index) => (
                <span key={index}>{line}</span>
              ))}
            </footer>
            {isExcerpt(match) && questions.length > 1 && (
              <div className="hit-questions">
                {questionLabels(match.aspects, questions, match.aspect_scores).map(({ label, question, score }) => (
                  <span key={label} className="question-tag" tabIndex={0} onClick={(event) => event.stopPropagation()}>
                    {label}
                    {score !== undefined && <span className="question-score"> {score}</span>}
                    <span className="hint hint-below hint-wide" role="tooltip">
                      <strong>
                        {label}
                        {score !== undefined && ` · ${score}`}
                      </strong>
                      <span>{question}</span>
                      {score !== undefined && <span className="muted">{questionScoreMeaning(label, reranked)}</span>}
                    </span>
                  </span>
                ))}
              </div>
            )}
          </div>
        </article>
      ))}
    </div>
  )
}
