import { Activity, Bot, ChartColumn, Compass, FileText, Library, Settings } from 'lucide-react'
import type { ComponentType, ReactNode } from 'react'
import logoSvg from '../../../design/haskie-logo.svg?raw'
import { href, type RouteName } from '../router'

// Same shape as `design.js` builds: the mark, then the wordmark, inside the link itself. The SVG
// is authored with `fill="currentColor"`, so it takes the colour of whatever holds it.
const LOGO_HTML = `${logoSvg}<span>haskie</span>`

export function Logo() {
  return <a className="logo" href={href({ name: 'explore' })} aria-label="Haskie" dangerouslySetInnerHTML={{ __html: LOGO_HTML }} />
}

/** Totals beside the nav labels; null while the first request is still out. */
export interface NavCounts {
  documents: number | null
  collections: number | null
}

interface NavEntry {
  name: RouteName
  label: string
  icon: ComponentType<{ className?: string }>
  href: string
  count?: number | null
}

export function Shell({
  current,
  counts,
  side,
  children,
}: {
  current: RouteName
  counts: NavCounts
  side?: ReactNode
  children: ReactNode
}) {
  const entries: NavEntry[] = [
    { name: 'explore', label: 'Explore', icon: Compass, href: href({ name: 'explore' }) },
    { name: 'documents', label: 'Documents', icon: FileText, href: href({ name: 'documents' }), count: counts.documents },
    { name: 'collections', label: 'Collections', icon: Library, href: href({ name: 'collections' }), count: counts.collections },
    { name: 'operations', label: 'Operations', icon: Activity, href: href({ name: 'operations' }) },
    { name: 'sessions', label: 'Sessions', icon: Bot, href: href({ name: 'sessions' }) },
    { name: 'insights', label: 'Insights', icon: ChartColumn, href: href({ name: 'insights' }) },
    { name: 'settings', label: 'Settings', icon: Settings, href: href({ name: 'settings' }) },
  ]

  return (
    <div className="page">
      <main className="body">
        <aside className="side">
          <Logo />
          <nav className="nav">
            {entries.map((entry) => (
              <a
                key={entry.name}
                className="nav-item"
                href={entry.href}
                aria-current={entry.name === current ? 'true' : undefined}
              >
                <entry.icon className="icon" />
                {entry.label}
                {entry.count !== null && entry.count !== undefined && <span className="count">{entry.count}</span>}
              </a>
            ))}
          </nav>
          {side}
        </aside>
        {children}
      </main>
    </div>
  )
}
