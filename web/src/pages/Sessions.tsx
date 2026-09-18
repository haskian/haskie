import { useEffect, useState } from 'react'
import { api } from '../api'
import { Search } from '../components/Search'

export function Sessions() {
  const [libraries, setLibraries] = useState<string[]>([])
  const [sessions, setSessions] = useState<Record<string, string[]>>({})
  const [current, setCurrent] = useState('')

  useEffect(() => {
    api.libraryNames().then(setLibraries)
    api.sessions().then(setSessions)
  }, [])

  const chosen = sessions[current] ?? []
  const toggle = async (lib: string) => {
    const next = chosen.includes(lib) ? chosen.filter((l) => l !== lib) : [...chosen, lib]
    await api.saveSession(current, next)
    setSessions({ ...sessions, [current]: next })
  }

  return (
    <div className="split">
      <aside>
        <h2>Sessions</h2>
        <ul className="list">
          {Object.keys(sessions).map((s) => (
            <li key={s}>
              <button className={s === current ? 'active' : ''} onClick={() => setCurrent(s)}>
                {s}
              </button>
            </li>
          ))}
        </ul>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            setSessions({ ...sessions, [current]: sessions[current] ?? [] })
          }}
        >
          <input placeholder="session id" value={current} onChange={(e) => setCurrent(e.target.value)} />
          <button disabled={!current.trim()}>Use</button>
        </form>
        <p className="muted">
          Agents call <code>set_session_libraries</code> then <code>search</code> with the same id.
        </p>
      </aside>
      <section>
        {current && (
          <>
            <h2>{current}</h2>
            <h3>Libraries searched</h3>
            {libraries.map((lib) => (
              <label key={lib} className="check">
                <input type="checkbox" checked={chosen.includes(lib)} onChange={() => toggle(lib)} /> {lib}
              </label>
            ))}
            <h3>Try a search</h3>
            <Search run={(q) => api.search(current, q)} libraries={chosen} />
          </>
        )}
      </section>
    </div>
  )
}
