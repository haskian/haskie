import { useCallback, useEffect, useState } from 'react'
import { api, type Status } from './api'
import { usePoll } from './hooks/usePoll'
import { Collections } from './pages/Collections'
import { Documents } from './pages/Documents'
import { Explore } from './pages/Explore'
import { Init } from './pages/Init'
import { Insights } from './pages/Insights'
import { Operations } from './pages/Operations'
import { Sessions } from './pages/Sessions'
import { Settings } from './pages/Settings'
import { applyBackground } from './pages/settings/background'
import { useRoute, type Route } from './router'
import { Statusbar, type NavCounts } from './ui'

const STATUS_POLL_MS = 5000

export default function App() {
  const route = useRoute()
  const [status, setStatus] = useState<Status | null>(null)
  const [counts, setCounts] = useState<NavCounts>({ documents: null, collections: null })

  const refresh = useCallback(
    () =>
      api
        .status()
        .then(setStatus)
        .catch(() => undefined),
    [],
  )
  useEffect(() => void refresh(), [refresh])
  // A model download is the only thing that changes the status on its own. A save that changes
  // which models are needed (the embedding profile, a reranker) asks for it through `refreshStatus`.
  const downloading = status?.models.some((model) => model.state === 'loading' || model.state === 'pending') ?? false
  usePoll(downloading, refresh, STATUS_POLL_MS)

  // Nav counts are re-read on every route change, rather than through a refresh context: the
  // pages that change a count are the pages you then navigate away from.
  useEffect(() => {
    Promise.all([api.documents({ page_size: 1 }), api.collections({ page_size: 1 })])
      .then(([documents, collections]) => setCounts({ documents: documents.total, collections: collections.total }))
      .catch(() => undefined)
  }, [route.name])

  // The background choice is a page-wide attribute that Settings writes as it is changed, so this
  // only has to apply what a previous visit stored.
  useEffect(applyBackground, [])

  if (status === null) return <div className="page" />
  if (!status.initialized) return <Init onDone={refresh} />

  return (
    <>
      <Page route={route} counts={counts} refreshStatus={refresh} />
      <Statusbar status={status} />
    </>
  )
}

/** What every page gets: its route, and the nav counts it hands to `Shell`. */
export interface PageProps<R extends Route = Route> {
  route: R
  counts: NavCounts
  refreshStatus: () => Promise<void> // re-read the status, after a save that changes the models it lists
}

// Each page renders `Shell` itself, so its side sections and content share one component's state.
function Page({ route, counts, refreshStatus }: PageProps) {
  switch (route.name) {
    case 'explore':
      return <Explore route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'documents':
      return <Documents route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'collections':
      return <Collections route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'operations':
      return <Operations route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'sessions':
      return <Sessions route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'insights':
      return <Insights route={route} counts={counts} refreshStatus={refreshStatus} />
    case 'settings':
      return <Settings route={route} counts={counts} refreshStatus={refreshStatus} />
  }
}
