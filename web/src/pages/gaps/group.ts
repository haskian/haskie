import type { GapQuestion, GapSignal, GapTopic, LoggedResult, ReplayedGap } from '../../api'
import { relative } from '../../format'
import type { RangeGroup } from '../../ui'

/** A topic's address: the id of its newest question, which names it. */
export const topicId = (topic: GapTopic): string => String(topic.questions[0].id)

/** A topic every question of which is only borderline: maybe answered, not a confirmed gap. */
export const isBorderline = (topic: GapTopic): boolean => topic.questions.every((question) => question.signal === 'borderline')

/** Three bands, each in the order the server ranked them: the confirmed gaps asked more than once,
 *  those asked once, then the topics that are only borderline, which the page folds away. */
export function bands(topics: GapTopic[]): (RangeGroup<GapTopic> & { collapsed: boolean })[] {
  const gaps = topics.filter((topic) => !isBorderline(topic))
  return [
    { label: 'Asked again', items: gaps.filter((topic) => topic.questions.length > 1), collapsed: false },
    { label: 'Asked once', items: gaps.filter((topic) => topic.questions.length === 1), collapsed: false },
    { label: 'Borderline', items: topics.filter(isBorderline), collapsed: true },
  ].filter((group) => group.items.length > 0)
}

const count = (n: number, one: string, many: string): string => `${n} ${n === 1 ? one : many}`

// Why a question counts as a gap, in the page's words (`search.gaps.Signal`).
const REASONS: Record<GapSignal, string> = {
  reported: 'the agent found no answer',
  empty: 'nothing came back',
  uncovered: 'no excerpt answers it',
  weak: 'weak match',
  borderline: 'maybe answered',
}

/** The line under a topic tile's question: how often, by how many sessions, how recently. */
export function topicSub(topic: GapTopic, now: number): string {
  const sessions = topic.sessions === 0 ? 'no session' : count(topic.sessions, 'session', 'sessions')
  return `${count(topic.questions.length, 'time', 'times')} · ${sessions} · ${relative(topic.last_at, now)}`
}

/** A tile's hint: why its questions count as gaps, and where they were looked for. */
export function topicHint(topic: GapTopic): string {
  const why = [...new Set(topic.questions.map((question) => REASONS[question.signal]))]
  const where = topic.collections.length === 0 ? ['no collections'] : topic.collections
  return [...why, ...where].join(' · ')
}

// A score as the page prints it: two decimals, and a real minus sign.
const score = (value: number): string => value.toFixed(2).replace('-', '−')

/** Why one question counts as a gap. For a weak match, the score that fell short: the reranker
 *  decides over the cosine when the search had one (`search.gaps`), so it is the one named. */
export function signalText(
  question: Pick<GapQuestion, 'best_rerank' | 'best_similarity'> & { signal: GapSignal | null; agent_verdict?: GapQuestion['agent_verdict']; agent_note?: GapQuestion['agent_note'] },
): string {
  if (question.signal === 'reported') {
    const verdict = question.agent_verdict === 'partial' ? 'the agent found a partial answer' : REASONS.reported
    return question.agent_note ? `${verdict}: ${question.agent_note}` : verdict
  }
  if (question.signal === 'borderline') {
    return question.best_similarity === null ? REASONS.borderline : `${REASONS.borderline}: cosine ${score(question.best_similarity)}`
  }
  if (question.signal !== 'weak') return question.signal === null ? 'answered' : REASONS[question.signal]
  if (question.best_rerank !== null) return `weak match: reranker ${score(question.best_rerank)}`
  if (question.best_similarity !== null) return `weak match: cosine ${score(question.best_similarity)}`
  return REASONS.weak
}

/** The line under a gap question: why, through which tool, for whom, when. */
export function questionSub(question: GapQuestion, now: number): string {
  return [signalText(question), `via ${question.tool}`, question.session_id ?? 'no session', relative(question.ts, now)].join(' · ')
}

/** What came closest to answering, by citation; null when nothing came back at all. */
export function nearText(results: LoggedResult[]): string | null {
  return results.length === 0 ? null : `closest: ${results.map((result) => result.location).join(' · ')}`
}

/** What a replay found: answered now, and where, every place it returned (the one that closed
 *  the gap need not rank first); or still a gap, and why. */
export function replayText(replayed: ReplayedGap): string {
  if (replayed.signal === null) {
    const where = replayed.results.map((result) => result.location)
    return where.length === 0 ? 'now answered' : `now answered: ${where.join(' · ')}`
  }
  return `now: still ${signalText(replayed)}`
}
