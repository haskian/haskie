import { useEffect, useState } from 'react'
import { api } from '../api'
import { Search } from '../components/Search'

// A session is the set of collections an agent searches under one id.
export function Sessions() {
  const [collections, setCollections] = useState<string[]>([])
  const [sessions, setSessions] = useState<Record<string, string[]>>({})
  const [current, setCurrent] = useState('')

  useEffect(() => {
    api.collectionNames().then(setCollections)
    api.sessions().then(setSessions)
  }, [])

  const chosen = sessions[current] ?? []
  const toggle = async (name: string) => {
    const next = chosen.includes(name) ? chosen.filter((c) => c !== name) : [...chosen, name]
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
          Agents call <code>set_session_collections</code> then <code>search</code> with the same id.
        </p>
      </aside>
      <section>
        {current && (
          <>
            <h2>{current}</h2>
            <h3>Collections searched</h3>
            {collections.map((name) => (
              <label key={name} className="check">
                <input type="checkbox" checked={chosen.includes(name)} onChange={() => toggle(name)} /> {name}
              </label>
            ))}
            <h3>Try a search</h3>
            <Search run={(q) => api.search(current, q)} collections={chosen} />
          </>
        )}
      </section>
    </div>
  )
}
