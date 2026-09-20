import { useState } from 'react'
import { api, type DocumentMatch, type Hit } from '../api'

// What a search is asked for: the passages that answer the query, or the documents to read for it.
type Mode = 'chunks' | 'documents'

export function Search({
  run,
  collections,
  placeholder = 'query',
}: {
  run: (q: string) => Promise<Hit[]>
  // The scope the document shortlist spans. Given, the mode dropdown appears; the chunk search
  // has its own scope already (one collection, or a session), which is why it stays a callback.
  collections?: string[]
  placeholder?: string
}) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<Mode>('chunks')
  const [hits, setHits] = useState<Hit[] | null>(null)
  const [documents, setDocuments] = useState<DocumentMatch[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const search = async () => {
    setHits(null)
    setDocuments(null)
    if (mode === 'documents') {
      // an empty scope is "no collections chosen", not "every collection": the server reads a
      // missing filter as all of them, so the request is not made at all
      setDocuments(collections?.length ? await api.searchDocuments(query, collections) : [])
    } else {
      setHits(await run(query))
    }
  }

  const results = mode === 'documents' ? documents : hits

  return (
    <>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          search()
            .then(() => setError(null))
            .catch((err) => setError(String(err)))
        }}
      >
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={placeholder} />
        {collections && (
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)} title="what to return">
            <option value="chunks">chunks</option>
            <option value="documents">documents</option>
          </select>
        )}
        <button disabled={!query.trim()}>Search</button>
      </form>
      {error && <p className="error">{error}</p>}
      {results && results.length === 0 && <p className="muted">no results</p>}
      {documents?.map((m) => (
        <article key={`${m.collection}/${m.doc}`} className="hit">
          <header className="muted">
            <strong>{m.collection} / {m.doc}</strong> {m.heading}
          </header>
          <div className="hit-meta muted">
            <span>score {m.score.toFixed(3)}</span>
            <span>{m.chunks} matching {m.chunks === 1 ? 'chunk' : 'chunks'}</span>
            {m.description && <span>{m.description}</span>}
          </div>
          <pre>{m.text}</pre>
        </article>
      ))}
      {hits?.map((h) => (
        <article key={`${h.collection}/${h.doc}/${h.chunk_id}`} className="hit">
          <header className="muted">
            <strong>{h.collection} / {h.location}</strong> {h.header}
          </header>
          <div className="hit-meta muted">
            {h.page_start !== null && <span>pages {h.page_start === h.page_end ? h.page_start : `${h.page_start}–${h.page_end}`}</span>}
            <span>lines {h.line_start}–{h.line_end}</span>
            <span>chars {h.char_start}–{h.char_end}</span>
            <span>chunk {h.chunk_id} · part {h.part}</span>
            <span>score {h.score.toFixed(3)}</span>
            <span title={h.markdown_file}>md: {h.markdown_path}</span>
            <span title={h.source_file}>src: {h.source_path}</span>
          </div>
          <pre>{h.text}</pre>
        </article>
      ))}
    </>
  )
}
