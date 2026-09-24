import { Library, Plus } from 'lucide-react'
import { useCallback, useMemo, useState, type FormEvent } from 'react'
import { api } from '../api'
import type { PageProps } from '../App'
import { usePaged } from '../hooks/usePaged'
import { usePoll } from '../hooks/usePoll'
import { navigate, type Route } from '../router'
import { Field, GallerySection, Modal, SearchBox, Shell, Tile } from '../ui'
import './Collections.css'
import { CollectionModal } from './collections/CollectionModal'
import { groupByName, tileSub } from './collections/group'
import { errorText, matchesText, needleOf } from '../format'

const PAGE_SIZE = 500
/** Every collection in the home, as a gallery banded by name; one modal per collection. */
export function Collections({ route, counts, refreshStatus }: PageProps<Extract<Route, { name: 'collections' }>>) {
  const [creating, setCreating] = useState(false)
  const [search, setSearch] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [createError, setCreateError] = useState<string | null>(null)

  const collections = usePaged((query) => api.collections(query), { sort: 'name', pageSize: PAGE_SIZE })
  const refresh = collections.refresh
  // indexing moves the counts under the page, so the listing follows it while anything is active
  const active = collections.items.some((one) => one.counts.active > 0)
  const poll = useCallback(() => {
    void refresh().catch(() => undefined)
  }, [refresh])
  usePoll(active, poll)

  // Client-side over the rows already loaded: a page of 500 is what the gallery shows anyway.
  const needle = needleOf(search)
  const groups = useMemo(() => groupByName(collections.items.filter((one) => matchesText(needle, one.name, one.description))), [collections.items, needle])

  const startCreating = () => {
    setName('')
    setDescription('')
    setCreateError(null)
    setCreating(true)
  }

  const create = async (event: FormEvent) => {
    event.preventDefault()
    try {
      const info = await api.createCollection(name.trim(), description.trim())
      setCreating(false)
      await refresh()
      navigate({ name: 'collections', collection: info.name })
    } catch (cause) {
      setCreateError(errorText(cause))
    }
  }

  const close = useCallback(() => navigate({ name: 'collections' }), [])
  // stable, because the modal's polls restart whenever it changes
  const changed = useCallback(() => Promise.all([refresh(), refreshStatus()]), [refresh, refreshStatus])

  return (
    <Shell current={route.name} counts={counts}>
      <div className="gallery-sections sections">
        <SearchBox id="search" value={search} onChange={setSearch} placeholder="Search collections" />
        {collections.error !== null && <p className="muted">{collections.error}</p>}
        <GallerySection label="New" large>
          <Tile icon={Plus} name="New collection" sub="Create one" hint="Name it, say what it holds." add onClick={startCreating} />
        </GallerySection>
        {groups.map((group) => (
          <GallerySection key={group.label} label={group.label} large>
            {group.items.map((one) => (
              <Tile
                key={one.name}
                icon={Library}
                name={one.name}
                sub={tileSub(one)}
                hint={one.description || 'No description'}
                onClick={() => navigate({ name: 'collections', collection: one.name })}
              />
            ))}
          </GallerySection>
        ))}
        {collections.hasMore && (
          <button className="btn btn-ghost" type="button" onClick={collections.loadMore}>
            Load more
          </button>
        )}
      </div>
      {/* a collection's reranker override is a model the status bar lists */}
      <CollectionModal name={route.collection} onClose={close} onChanged={changed} />
      <Modal open={creating} onClose={() => setCreating(false)} title="New collection" subtitle="collection">
        {/* A form, so Enter creates the way the browser already does it. */}
        <form className="collection-panel" onSubmit={create}>
          <Field label="Name">
            <input className="input" value={name} placeholder="Collection name" autoFocus onChange={(event) => setName(event.target.value)} />
          </Field>
          <Field label="Description">
            <textarea className="textarea" rows={4} value={description} placeholder="What it holds" onChange={(event) => setDescription(event.target.value)} />
          </Field>
          <div className="row">
            <button className="btn btn-primary" type="submit" disabled={name.trim() === ''}>
              Create
            </button>
          </div>
          {createError !== null && <p className="muted collection-error">{createError}</p>}
        </form>
      </Modal>
    </Shell>
  )
}
