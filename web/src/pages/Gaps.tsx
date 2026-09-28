import { Check, RotateCcw, SearchX, Undo2, X } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { api, type GapReview, type GapSignal, type GapTopic, type ReplayedGap } from '../api'
import type { PageProps } from '../App'
import { errorText } from '../format'
import { useRun } from '../hooks/useRun'
import { navigate, type Route } from '../router'
import { GallerySection, Modal, Shell, Tabs, Tile, type TabDef } from '../ui'
import './Gaps.css'
import { bands, nearText, questionSub, replayText, topicHint, topicId, topicSub } from './gaps/group'

// The tab id is the review state the list is read with.
const TABS: TabDef[] = [
  { id: 'open', label: 'Open' },
  { id: 'dismissed', label: 'Dismissed' },
  { id: 'resolved', label: 'Resolved' },
]
const DAYS = 30
// every reason, the borderline one included: the page folds those away rather than hiding them
const SIGNALS: GapSignal[] = ['reported', 'empty', 'uncovered', 'weak', 'borderline']
const EMPTY: Record<GapReview, string> = {
  open: `No gaps. Every question of the last ${DAYS} days found an answer.`,
  dismissed: 'Nothing dismissed.',
  resolved: 'Nothing resolved.',
}
const NOTHING_TO_REFRESH = async (): Promise<void> => undefined
const MAX_REPLAY = 50 // the backend's cap on one replay; a bigger topic replays its newest questions

/** The questions the shelf did not answer: one tile per topic, the most asked first, one modal per topic. */
export function Gaps({ route, counts }: PageProps<Extract<Route, { name: 'gaps' }>>) {
  const [review, setReview] = useState<GapReview>('open')
  const [loaded, setLoaded] = useState<{ topics: GapTopic[]; now: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  // "2 h ago" is measured from the moment the list was read
  const refresh = useCallback(() => api.gaps(review, DAYS, SIGNALS).then((topics) => setLoaded({ topics, now: Date.now() / 1000 })), [review])
  useEffect(() => {
    refresh().catch((cause: unknown) => setError(errorText(cause)))
  }, [refresh])

  const close = useCallback(() => navigate({ name: 'gaps' }), [])
  const open = loaded?.topics.find((topic) => topicId(topic) === route.topic)
  const reviewed = useCallback(() => {
    close()
    refresh().catch((cause: unknown) => setError(errorText(cause)))
  }, [close, refresh])

  return (
    <Shell current={route.name} counts={counts}>
      <div className="gallery-sections sections">
        <Tabs tabs={TABS} selected={review} onSelect={(id) => setReview(id as GapReview)} />
        {error !== null && <p className="muted">{error}</p>}
        {loaded !== null && loaded.topics.length === 0 && <p className="muted">{EMPTY[review]}</p>}
        {loaded !== null &&
          bands(loaded.topics).map((group) => (
            <GallerySection key={group.label} label={`${group.label} · ${group.items.length}`} large collapsed={group.collapsed}>
              {group.items.map((topic) => (
                <Tile
                  key={topicId(topic)}
                  icon={SearchX}
                  name={topic.question}
                  sub={topicSub(topic, loaded.now)}
                  hint={topicHint(topic)}
                  onClick={() => navigate({ name: 'gaps', topic: topicId(topic) })}
                />
              ))}
            </GallerySection>
          ))}
      </div>
      <Modal open={open !== undefined} onClose={close} title={open?.question ?? ''} subtitle="gap">
        {open !== undefined && loaded !== null && <TopicBody key={topicId(open)} topic={open} review={review} now={loaded.now} onReviewed={reviewed} />}
      </Modal>
    </Shell>
  )
}

/** One topic: its questions with what came closest, a replay against the shelf as it is now, and the curator's decision. */
function TopicBody({ topic, review, now, onReviewed }: { topic: GapTopic; review: GapReview; now: number; onReviewed: () => void }) {
  const [replayed, setReplayed] = useState<Map<number, ReplayedGap> | null>(null)
  // nothing to re-read after a replay; a decision closes the modal and re-reads the page itself
  const { run, busy, error } = useRun(NOTHING_TO_REFRESH)
  const ids = topic.questions.map((question) => question.id)

  const replay = () =>
    run(async () => {
      const found = await api.replayGaps(ids.slice(0, MAX_REPLAY))
      setReplayed(new Map(found.map((one) => [one.id, one])))
    })
  const decide = (to: GapReview) =>
    run(async () => {
      await api.reviewGaps(ids, to)
      onReviewed()
    })

  return (
    <>
      <div className="row">
        <button className="btn btn-mono" type="button" disabled={busy} onClick={replay}>
          <RotateCcw className="icon" />
          Replay
        </button>
        {review === 'open' ? (
          <>
            <button className="btn btn-mono" type="button" disabled={busy} onClick={() => decide('resolved')}>
              <Check className="icon" />
              Resolve
            </button>
            <button className="btn btn-mono" type="button" disabled={busy} onClick={() => decide('dismissed')}>
              <X className="icon" />
              Dismiss
            </button>
          </>
        ) : (
          <button className="btn btn-mono" type="button" disabled={busy} onClick={() => decide('open')}>
            <Undo2 className="icon" />
            Reopen
          </button>
        )}
      </div>
      <div role="tabpanel" className="gap-panel">
        {error !== null && <p className="muted">{error}</p>}
        <p className="muted">Replay asks each question again over every collection. Nothing is recorded.</p>
        <ul className="list">
          {topic.questions.map((question) => {
            const near = nearText(question.near_misses)
            const again = replayed?.get(question.id)
            return (
              <li className="list-item" key={question.id}>
                <SearchX className="icon" />
                <span className="list-text">
                  <span className="gap-query">{question.question}</span>
                  <span className="sub">{questionSub(question, now)}</span>
                  {near !== null && <span className="sub gap-near">{near}</span>}
                  {again !== undefined && <span className="sub gap-now">{replayText(again)}</span>}
                </span>
              </li>
            )
          })}
        </ul>
      </div>
    </>
  )
}
