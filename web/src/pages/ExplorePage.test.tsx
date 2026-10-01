import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { Explore } from './Explore'

describe('Explore', () => {
  test('the debug switch sits on the line over the results, off, with no raw response yet', () => {
    const html = renderToStaticMarkup(<Explore route={{ name: 'explore' }} counts={{ documents: 3, collections: 1 }} refreshStatus={async () => {}} />)

    expect(html).toMatch(/<div class="explore-status"><p class="mono muted search-took"[^]*<label class="toggle"><input type="checkbox" role="switch"\/>Debug<\/label><\/div>/)
    expect(html).not.toContain('explore-raw')
  })
})
