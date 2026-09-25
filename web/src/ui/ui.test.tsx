import { describe, expect, test } from 'bun:test'
import { FileText } from 'lucide-react'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { EmbedderMetadata, Hit, Passage, RerankerMetadata, Source, Status } from '../api'
import { Field } from './Field'
import { GallerySection } from './GallerySection'
import { HitGrid } from './HitGrid'
import { Kv } from './Kv'
import { MatchModal } from './MatchModal'
import { ModelFacts } from './ModelFacts'
import { Mark } from './Mark'
import { markTerms } from './markTerms'
import { Picker } from './Picker'
import { SearchTook } from './SearchTook'
import { Statusbar } from './Statusbar'
import { Jobs, type JobBar } from './Jobs'
import { Tabs } from './Tabs'
import { Tile } from './Tile'
import { Check, Toggle } from './Toggle'

interface MarkupCase {
  name: string
  element: ReactElement
  contains: string[]
  missing?: string[]
}

function check(cases: MarkupCase[]): void {
  for (const markupCase of cases) {
    test(markupCase.name, () => {
      const html = renderToStaticMarkup(markupCase.element)
      for (const needle of markupCase.contains) expect(html).toContain(needle)
      for (const needle of markupCase.missing ?? []) expect(html).not.toContain(needle)
    })
  }
}

const noop = (): void => {}

const HIT: Hit = {
  collection: 'A–E',
  document: 'area.pdf',
  source_path: 'sources/area.pdf',
  markdown_path: 'markdown/area.md',
  part: 0,
  seq: 4,
  line_start: 12,
  line_end: 18,
  char_start: 420,
  char_end: 640,
  byte_start: 420,
  byte_end: 640,
  page_start: 2,
  page_end: 2,
  headings: ['Lighting', 'Soft shadows'],
  frame: ['Lighting', 'Soft shadows'],
  header: 'Lighting › Soft shadows',
  location: 'p. 2',
  text: 'Area lights soften the shadow edge in proportion to their size.',
  layout: [{ type: 'text', position: 0 }],
  start_reason: 'paragraph',
  end_reason: 'length_sentence',
  score: 0.9123,
  source_file: '/home/ada/.haskie/sources/area.pdf',
  markdown_file: '/home/ada/.haskie/markdown/area.md',
  also_in: [],
}

const SOURCE: Source = {
  collection: 'P–T',
  document: 'sun.pdf',
  score: 0.79,
  chunks: 6,
  description: 'Sun position by date, time and latitude.',
  header: 'Solar geometry > Elevation tables',
  location: 'p. 4',
  text: 'Sun position at 35° elevation casts a shadow 1.4× the object height.',
  source_file: '/home/ada/.haskie/sources/sun.pdf',
  markdown_file: '/home/ada/.haskie/markdown/sun.md',
  line_start: 3,
  line_end: 9,
  collections: ['P–T', 'U–Z'],
  sections: [
    { header: 'Elevation tables', score: 0.79, chunks: 4, line_start: 3, line_end: 9, location: 'sun.pdf p.4 L3-9' },
    { header: 'Elevation tables > Corrections', score: 0.31, chunks: 2, line_start: 12, line_end: 15, location: 'sun.pdf p.5 L12-15' },
  ],
}

const PASSAGE: Passage = {
  collection: 'A–E',
  document: 'area.pdf',
  header: 'Lighting > Soft shadows',
  location: 'area.pdf p.2 L41-58',
  seq_start: 4,
  seq_end: 5,
  line_start: 41,
  line_end: 58,
  char_start: 1204,
  char_end: 2102,
  page_start: 2,
  page_end: 2,
  text: 'Area lights soften the shadow edge in proportion to their size. A larger light reads as a softer edge.',
  score: 0.88,
  source_file: '/home/ada/.haskie/sources/area.pdf',
  markdown_file: '/home/ada/.haskie/markdown/area.md',
  also_in: [],
}

const bar = (over: Partial<JobBar> = {}): JobBar => ({ label: 'Embed', done: 9, total: 22, state: 'active', ...over })

describe('Picker', () => {
  const options = [
    { value: '*', label: 'All collections', sub: '30 documents' },
    { value: 'A–E', label: 'A–E' },
  ]
  check([
    {
      name: 'summary shows the selected label and sub',
      element: <Picker options={options} value="*" onChange={noop} />,
      contains: ['<details class="picker"', '<span class="picker-value"><span>All collections</span><span class="sub">30 documents</span>'],
    },
    {
      name: 'an option without a sub renders no sub span in the summary',
      element: <Picker options={options} value="A–E" onChange={noop} />,
      contains: ['<span class="picker-value"><span>A–E</span></span>'],
    },
    {
      name: 'the list is a listbox of focusable options',
      element: <Picker options={options} value="*" onChange={noop} />,
      contains: ['<ul class="picker-list" role="listbox"', 'role="option" tabindex="0"'],
    },
    {
      name: 'only the selected option is aria-selected',
      element: <Picker options={options} value="A–E" onChange={noop} />,
      contains: ['aria-selected="false"><span>All collections</span>', 'aria-selected="true"><span>A–E</span>'],
    },
    {
      name: 'a value no option carries leaves the summary empty',
      element: <Picker options={options} value="gone" onChange={noop} />,
      contains: ['<span class="picker-value"><span></span></span>'],
      missing: ['aria-selected="true"'],
    },
    {
      name: 'aria-label reaches the summary and the listbox',
      element: <Picker options={options} value="*" onChange={noop} ariaLabel="Scope" />,
      contains: ['<summary aria-label="Scope">', 'role="listbox" aria-label="Scope"'],
    },
    {
      name: 'no options renders an empty list',
      element: <Picker<string> options={[]} value="*" onChange={noop} />,
      contains: ['<ul class="picker-list" role="listbox"></ul>'],
    },
  ])
})

describe('Tabs', () => {
  const tabs = [
    { id: 'tab-matches', label: 'Excerpts · 6' },
    { id: 'tab-sources', label: 'Sources · 3' },
  ]
  check([
    {
      name: 'the strip is a tablist',
      element: <Tabs tabs={tabs} selected="tab-matches" onSelect={noop} />,
      contains: ['<div class="tabs" role="tablist">'],
    },
    {
      name: 'the selected tab is the only one marked',
      element: <Tabs tabs={tabs} selected="tab-sources" onSelect={noop} />,
      contains: ['aria-selected="false" aria-controls="tab-matches">Excerpts · 6', 'aria-selected="true" aria-controls="tab-sources">Sources · 3'],
    },
    {
      name: 'no tabs renders an empty strip',
      element: <Tabs tabs={[]} selected="" onSelect={noop} />,
      contains: ['<div class="tabs" role="tablist"></div>'],
    },
  ])
})

describe('Tile', () => {
  check([
    {
      name: 'icon, text, sub and hint',
      element: <Tile icon={FileText} name="Area" sub="A. Hoffmann" hint="Notes on area lights." onClick={noop} />,
      contains: [
        '<button class="tile" type="button"',
        'lucide-file-text icon"',
        '<span class="tile-text"><span class="name">Area</span><span class="sub">A. Hoffmann</span></span>',
        '<span class="hint" role="tooltip"><strong>Area</strong><span class="sub">A. Hoffmann</span>Notes on area lights.</span>',
      ],
    },
    {
      name: 'pressed sets aria-pressed',
      element: <Tile icon={FileText} name="Area" sub="" hint="" pressed onClick={noop} />,
      contains: ['aria-pressed="true"'],
    },
    {
      name: 'pressed omitted leaves a plain button',
      element: <Tile icon={FileText} name="Area" sub="" hint="" onClick={noop} />,
      contains: ['class="tile"'],
      missing: ['aria-pressed'],
    },
  ])
})

describe('GallerySection', () => {
  check([
    {
      name: 'label and grid',
      element: <GallerySection label="A–E">x</GallerySection>,
      contains: ['<section class="gallery-section section">', '<span class="mono muted">A–E</span>', '<div class="gallery">x</div>'],
    },
    { name: 'large swaps in the wide grid', element: <GallerySection label="A–E" large>{null}</GallerySection>, contains: ['class="gallery gallery-lg"'] },
    { name: 'id reaches the section', element: <GallerySection label="A–E" id="g-a-e">{null}</GallerySection>, contains: ['id="g-a-e"'] },
  ])
})

describe('Jobs', () => {
  check([
    {
      name: 'glass with stripes',
      element: <Jobs jobs={[bar()]} variant="glass" stripes />,
      contains: ['class="stages stages-glass stages-stripes"'],
    },
    {
      name: 'line variant drops the stripes',
      element: <Jobs jobs={[bar({ state: 'done', done: 22 })]} variant="line" />,
      contains: ['class="stages stages-line"'],
      missing: ['stages-stripes'],
    },
    {
      name: 'an active job spins the settings icon and fills the bar to its share',
      element: <Jobs jobs={[bar({ weight: 2.2 })]} variant="glass" />,
      contains: [
        'class="stage active"',
        '--progress:0.4090909090909091',
        '--weight:2.2',
        '<span>Embed</span>',
        'lucide-settings icon spin"',
        '<span class="stage-bar"></span>',
        '<span class="stage-meta"><span>9/22</span><span></span></span>',
      ],
    },
    {
      name: 'a done job checks off and fills the bar',
      element: <Jobs jobs={[bar({ state: 'done', done: 22, seconds: 38 })]} variant="line" />,
      contains: ['class="stage done"', '--progress:1', 'lucide-check icon"', '<span>22/22</span>', '<span>38 sec</span>'],
    },
    {
      name: 'a todo job is plain, with a clock and an empty bar',
      element: <Jobs jobs={[bar({ state: 'todo', done: 0 })]} variant="line" />,
      contains: ['class="stage"', '--progress:0', 'lucide-clock icon"'],
      missing: ['spin'],
    },
    {
      name: 'an error job is plain, with a cross',
      element: <Jobs jobs={[bar({ state: 'error', done: 0 })]} variant="line" />,
      contains: ['class="stage"', 'lucide-x icon"'],
      missing: ['stage done', 'stage active'],
    },
    {
      name: 'a job with no total and nothing to say knows no counts, so it shows none',
      element: <Jobs jobs={[bar({ state: 'todo', done: 0, total: 0 })]} variant="line" />,
      contains: ['--progress:0'],
      missing: ['stage-meta'],
    },
    {
      name: 'a done job with no total fills its bar anyway',
      element: <Jobs jobs={[bar({ state: 'done', done: 0, total: 0 })]} variant="line" />,
      contains: ['class="stage done"', '--progress:1'],
      missing: ['stage-meta'],
    },
    {
      name: 'a note stands in for the time a job was never timed at',
      element: <Jobs jobs={[bar({ state: 'done', done: 0, total: 0, note: 'loaded' })]} variant="line" />,
      contains: ['<span class="stage-meta"><span></span><span>loaded</span></span>'],
      missing: ['0/0'],
    },
    {
      name: 'a note follows the counts when the job has both',
      element: <Jobs jobs={[bar({ state: 'done', done: 22, note: 'cached' })]} variant="glass" />,
      contains: ['<span class="stage-meta"><span>22/22</span><span>cached</span></span>'],
    },
    {
      name: 'a timed job says its note and its duration together',
      element: <Jobs jobs={[bar({ state: 'done', done: 22, seconds: 38, note: 'loaded' })]} variant="line" />,
      contains: ['<span>22/22</span>', '<span>loaded · 38 sec</span>'],
    },
    {
      name: 'no weight leaves the custom property out',
      element: <Jobs jobs={[bar()]} variant="glass" />,
      contains: ['--progress:'],
      missing: ['--weight'],
    },
    { name: 'no jobs renders an empty strip', element: <Jobs jobs={[]} variant="line" />, contains: ['class="stages stages-line"'], missing: ['stage-bar'] },
  ])
})

describe('Kv', () => {
  check([
    {
      name: 'a row per pair',
      element: <Kv rows={[['Score', '0.91'], ['Collection', 'A–E']]} />,
      contains: ['<dl class="kv"><dt>Score</dt><dd>0.91</dd><dt>Collection</dt><dd>A–E</dd></dl>'],
    },
    {
      name: 'a value may be an element',
      element: <Kv rows={[['Query', <span key="q" className="code">shadow</span>]]} />,
      contains: ['<dd><span class="code">shadow</span></dd>'],
    },
    { name: 'no rows renders an empty list', element: <Kv rows={[]} />, contains: ['<dl class="kv"></dl>'] },
  ])
})

// as `/api/options` sends them: bge-small under `embedding_metadata`, jina v3 under `reranker_metadata`
const EMBEDDER: EmbedderMetadata = {
  description: 'Small and fast; a good default (~130 MB).',
  parameters: 33360512,
  context_tokens: 512,
  languages: 'English',
  license: 'MIT',
  released: '2023-09-12',
  model_card_url: 'https://huggingface.co/BAAI/bge-small-en-v1.5',
  runtime: 'onnx',
  devices: ['cpu', 'apple_silicon', 'gpu'],
  dimensions: 384,
}
const RERANKER: RerankerMetadata = {
  description: 'Listwise: reads every candidate together and ranks them against each other.',
  parameters: 596836352,
  context_tokens: 131072,
  languages: 'multilingual',
  license: 'CC BY-NC 4.0 (non-commercial)',
  released: '2025-09-18',
  model_card_url: 'https://huggingface.co/jinaai/jina-reranker-v3',
  runtime: 'mlx',
  devices: ['apple_silicon'],
}

describe('ModelFacts', () => {
  check([
    {
      name: 'an embedder: its vector size first, then every fact in order',
      element: <ModelFacts name="BAAI/bge-small-en-v1.5" metadata={EMBEDDER} />,
      contains: [
        '<dt>Model</dt><dd>BAAI/bge-small-en-v1.5</dd><dt>Dimensions</dt><dd>384</dd><dt>Parameters</dt><dd>33M</dd>' +
          '<dt>Context</dt><dd>512 tokens</dd><dt>Released</dt><dd>2023-09-12</dd>' +
          '<dt>Languages</dt><dd>English</dd><dt>License</dt><dd>MIT</dd><dt>Runtime</dt><dd>ONNX</dd>' +
          '<dt>Devices</dt><dd>CPU, Apple Silicon, GPU (the gpu extra)</dd>' +
          '<dt>Model card</dt><dd><a href="https://huggingface.co/BAAI/bge-small-en-v1.5" target="_blank" rel="noreferrer">' +
          'huggingface.co/BAAI/bge-small-en-v1.5</a></dd>',
      ],
    },
    {
      name: 'an MLX reranker: no vector size, the extra it needs, and the card it was made from',
      element: <ModelFacts name="jinaai/jina-reranker-v3-mlx" metadata={RERANKER} />,
      contains: [
        '<dt>Parameters</dt><dd>597M</dd><dt>Context</dt><dd>131K tokens</dd>',
        '<dt>Runtime</dt><dd>MLX (the mlx extra)</dd><dt>Devices</dt><dd>Apple Silicon</dd>',
        'href="https://huggingface.co/jinaai/jina-reranker-v3"',
      ],
      missing: ['Dimensions'],
    },
    {
      name: 'metadata missing: the name alone',
      element: <ModelFacts name="x/gone" metadata={undefined} />,
      contains: ['<dl class="kv"><dt>Model</dt><dd>x/gone</dd></dl>'],
    },
  ])
})

describe('Field', () => {
  check([
    {
      name: 'label then control',
      element: <Field label="Chunk size"><input className="input" /></Field>,
      contains: ['<div class="field"><span class="label">Chunk size</span><input class="input"'],
      missing: ['faint'],
    },
    {
      name: 'help renders under the control',
      element: <Field label="Chunk size" help="Tokens per chunk."><input className="input" /></Field>,
      contains: ['<span class="faint">Tokens per chunk.</span>'],
    },
    {
      name: 'empty help renders nothing',
      element: <Field label="Chunk size" help=""><input className="input" /></Field>,
      contains: ['class="field"'],
      missing: ['faint'],
    },
  ])
})

describe('Toggle and Check', () => {
  check([
    {
      name: 'a checked toggle is a switch',
      element: <Toggle label="Classic background" checked onChange={noop} />,
      contains: ['<label class="toggle">', 'role="switch"', 'checked=""', '>Classic background</label>'],
    },
    {
      name: 'an unchecked toggle is not checked',
      element: <Toggle label="Skip OCR pages" checked={false} onChange={noop} />,
      contains: ['role="switch"'],
      missing: ['checked'],
    },
    { name: 'id reaches the input', element: <Toggle label="Classic" checked onChange={noop} id="bg-classic" />, contains: ['id="bg-classic"'] },
    { name: 'a toggle may be disabled', element: <Toggle label="A" checked={false} disabled onChange={noop} />, contains: ['disabled=""'] },
    {
      name: 'a check is a plain checkbox',
      element: <Check label="A–E" checked onChange={noop} />,
      contains: ['<label class="check">', 'type="checkbox"', 'checked=""', '>A–E</label>'],
      missing: ['role="switch"'],
    },
  ])
})

describe('markTerms', () => {
  const cases: Array<{ name: string; text: string; query: string; expected: string }> = [
    { name: 'no query leaves the text alone', text: 'soft shadow edge', query: '', expected: 'soft shadow edge' },
    { name: 'whitespace-only query leaves the text alone', text: 'soft shadow edge', query: '   ', expected: 'soft shadow edge' },
    { name: 'one term', text: 'soft shadow edge', query: 'shadow', expected: 'soft <mark>shadow</mark> edge' },
    { name: 'every occurrence', text: 'shadow on shadow', query: 'shadow', expected: '<mark>shadow</mark> on <mark>shadow</mark>' },
    { name: 'case is ignored, the text keeps its own', text: 'Shadow edge', query: 'shadow', expected: '<mark>Shadow</mark> edge' },
    { name: 'each term of a multi-word query', text: 'soft shadow edge', query: 'soft edge', expected: '<mark>soft</mark> shadow <mark>edge</mark>' },
    { name: 'a term matches inside a word', text: 'shadows', query: 'shadow', expected: '<mark>shadow</mark>s' },
    { name: 'no match leaves the text alone', text: 'soft edge', query: 'shadow', expected: 'soft edge' },
    { name: 'regex metacharacters are literal', text: 'a.b and axb', query: 'a.b', expected: '<mark>a.b</mark> and axb' },
    { name: 'empty text', text: '', query: 'shadow', expected: '' },
  ]
  for (const testCase of cases) {
    test(testCase.name, () => {
      expect(renderToStaticMarkup(<Mark text={testCase.text} query={testCase.query} />)).toBe(testCase.expected)
    })
  }

  test('returns the text unsplit when there is no query', () => {
    expect(markTerms('soft shadow edge', '')).toEqual(['soft shadow edge'])
  })
})

describe('HitGrid', () => {
  check([
    {
      name: 'a chunk hit carries its collection, score, marked text and position',
      element: <HitGrid results={[HIT]} query="shadow" />,
      contains: [
        '<div class="hits">',
        'class="hit" style="--score:1"',
        '<span class="tag"><span class="kind">A–E</span><span>area.pdf</span></span>',
        '<span class="mono muted">0.91</span>',
        '<mark>shadow</mark>',
        '<span>Soft shadows</span><span>p. 2</span><span>chunk 4</span>',
      ],
    },
    {
      name: 'a hit with no heading falls back to the header',
      element: <HitGrid results={[{ ...HIT, headings: [] }]} query="shadow" />,
      contains: ['<span>Lighting › Soft shadows</span>'],
    },
    {
      name: 'a hit with no page shows the chunk alone',
      element: <HitGrid results={[{ ...HIT, page_start: null }]} query="shadow" />,
      contains: ['chunk '],
      missing: ['p. '],
    },
    {
      name: 'the bar ranks a hit among the others: best full, worst at the floor',
      element: <HitGrid results={[HIT, { ...HIT, seq: 4, score: 0.4 }]} query="shadow" />,
      contains: ['style="--score:1"', 'style="--score:0.1"'],
    },
    {
      name: 'a source carries the same tag head, its chunk count and its section count',
      element: <HitGrid results={[SOURCE]} query="shadow" />,
      contains: [
        '<span class="kind">P–T</span><span>sun.pdf</span>',
        'Sun position by date, time and latitude.',
        '<span>6 chunks · 2 sections</span>',
      ],
      missing: ['hit-title'],
    },
    {
      name: 'a passage: its heading on one line, its page and lines on the next',
      element: <HitGrid results={[PASSAGE]} query="shadow" />,
      contains: ['<footer class="hit-foot"><span>Soft shadows</span><span>p. 2 · lines 41–58</span><span>chunks 4–5</span></footer>'],
    },
    {
      name: 'a passage without pages is its lines alone',
      element: <HitGrid results={[{ ...PASSAGE, page_start: null, page_end: null }]} query="shadow" />,
      contains: ['<span>lines 41–58</span><span>chunks 4–5</span>'],
    },
    {
      name: 'a chunk with no heading keeps the heading line, empty',
      element: <HitGrid results={[{ ...HIT, headings: [], header: '' }]} query="shadow" />,
      contains: ['<footer class="hit-foot"><span>\u00a0</span><span>p. 2</span><span>chunk 4</span></footer>'],
    },
    {
      name: 'a source without a description falls back to the matched text',
      element: <HitGrid results={[{ ...SOURCE, description: '' }]} query="shadow" />,
      contains: ['<mark>shadow</mark> 1.4×'],
    },
    { name: 'no hits renders an empty grid', element: <HitGrid results={[] as Hit[]} query="" />, contains: ['<div class="hits"></div>'] },
    { name: 'no sources renders an empty grid', element: <HitGrid results={[] as Source[]} query="" />, contains: ['<div class="hits"></div>'] },
  ])
})

describe('MatchModal', () => {
  check([
    {
      name: 'the heading path it was embedded under is grey at the top, its pieces white, each with a hint',
      element: (
        <MatchModal
          match={{
            ...HIT,
            text: '- Area lights soften it.\n\n| a |\n|---|',
            layout: [
              { type: 'list', position: 0 },
              { type: 'table', position: 26 },
            ],
          }}
          query=""
          onClose={noop}
        />
      ),
      contains: [
        '<blockquote class="match-text chunk-text"><span class="chunk-piece chunk-frame">Lighting &gt; Soft shadows\n\n<span class="hint" role="tooltip"><strong>Heading path</strong><span class="sub mono">prepended to the chunk when it was embedded</span></span></span>',
        '<span class="chunk-piece">- Area lights soften it.\n\n<span class="hint" role="tooltip"><strong>List item</strong><span class="sub mono">position 0 · 26 chars · 4 words</span></span></span>',
        '<span class="chunk-piece">| a |\n|---|<span class="hint" role="tooltip"><strong>Table</strong>',
      ],
    },
    {
      name: 'a chunk before any heading has no grey path',
      element: <MatchModal match={{ ...HIT, headings: [], frame: [], header: '' }} query="" onClose={noop} />,
      contains: ['<blockquote class="match-text chunk-text"><span class="chunk-piece">'],
    },
    {
      name: 'a passage keeps its heading row, and has no heading path row',
      element: <MatchModal match={PASSAGE} query="shadow" onClose={noop} />,
      contains: ['<dt>Heading</dt><dd>Soft shadows</dd><dt>Query</dt>'],
    },
    {
      name: 'a chunk names the rule that cut it on each side, above and below its text',
      element: <MatchModal match={{ ...HIT, start_reason: 'heading', end_reason: 'length_sentence' }} query="shadow" onClose={noop} />,
      contains: [
        'cut before: <span class="chunk-cut-reason">heading</span> · a heading starts a new section',
        'cut after: <span class="chunk-cut-reason">length_sentence</span> · chunk full, cut between two sentences',
        '<blockquote class="match-text chunk-text">',
        '<dt>Position</dt><dd>p. 2 · chunk 4</dd><dt>Query</dt><dd><span class="code">shadow</span></dd></dl>',
        '<span class="chunk-piece">Area lights soften the <mark>shadow</mark> edge in proportion to their size.<span class="hint" role="tooltip"><strong>Text</strong><span class="sub mono">position 0 · 63 chars · 11 words</span></span></span>',
        '<th>frame</th>',
        '<th>text</th>',
        '<th>total</th>',
      ],
    },
  ])
})

describe('also_in', () => {
  // the same paragraph in a second book, folded into the result by the search
  const REFERENCE = {
    collection: 'A–E',
    document: 'lighting-notes.md',
    header: 'Shadows > Area lights',
    location: 'lighting-notes.md L12-14',
    line_start: 12,
    line_end: 14,
    score: 0.74,
    relation: 'contained' as const,
    similarity: 0.97,
    // measured as `/api/search/excerpts` sends it: by words, by vectors, no shared characters
    to_parent: {
      words: { contained: 0.97, contains: 0.41, alike: 0.38, score: 0.5764 },
      embedding: { contained: 0.95, contains: 0.9, alike: 0.93, score: 0.9243 },
      chars: null,
    },
    to_root: {
      words: { contained: 0.97, contains: 0.41, alike: 0.38, score: 0.5764 },
      embedding: { contained: 0.95, contains: 0.9, alike: 0.93, score: 0.9243 },
      chars: null,
    },
    also_in: [],
  }
  check([
    {
      name: 'a tile counts the places that say the same, and the other documents they are in',
      element: (
        <HitGrid
          results={[
            {
              ...PASSAGE,
              also_in: [
                { ...REFERENCE, seq_start: 3, seq_end: 3 },
                { ...REFERENCE, seq_start: 9, seq_end: 9 },
                { ...REFERENCE, document: 'render-book.pdf', seq_start: 2, seq_end: 2 },
                { ...REFERENCE, document: PASSAGE.document, seq_start: 40, seq_end: 40 },
              ],
            },
          ]}
          query="shadow"
        />
      ),
      contains: [' · also in 4 / 2</span>'],
    },
    {
      name: 'a tile with nothing folded says nothing about it',
      element: <HitGrid results={[HIT]} query="shadow" />,
      missing: ['also in'],
      contains: [],
    },
    {
      name: 'the modal lists each place, how close it is and where, and counts them all',
      element: <MatchModal match={{ ...HIT, also_in: [{ ...REFERENCE, seq: 3 }, { ...REFERENCE, seq: 8, location: 'lighting-notes.md L30-31' }] }} query="shadow" onClose={noop} />,
      contains: [
        '<div class="sections-head"><span>Also in</span><span class="mono muted">2 places</span></div>',
        'inside 0.97</span>',
        '<span class="section-title">lighting-notes.md · Shadows &gt; Area lights</span>',
        '<span class="mono muted">L12-14</span>',
      ],
    },
    {
      name: 'a place folded into another sits indented under it, and the count takes every level',
      element: (
        <MatchModal
          match={{
            ...HIT,
            also_in: [
              {
                ...REFERENCE,
                seq: 3,
                also_in: [
                  {
                    ...REFERENCE,
                    seq: 8,
                    document: 'notes.md',
                    header: 'Delivery',
                    location: 'notes.md L4-5',
                    relation: 'equivalent' as const,
                    similarity: 0.99,
                  },
                ],
              },
              { ...REFERENCE, seq: 9, header: 'Shadows › Second cut' },
            ],
          }}
          query="shadow"
          onClose={noop}
        />
      ),
      contains: [
        '<span class="mono muted">3 places</span>',
        '<div class="also-nested"><details><summary class="section-row"><span class="mono muted" title="equivalent, query 0.74',
        'same meaning 0.99</span><span class="section-title">notes.md · Delivery</span>',
        '</div><details><summary class="section-row"><span class="mono muted" title="contained, query 0.74',
      ],
    },
    {
      name: 'each place is a closed disclosure: its lines are read only when it is opened',
      element: <MatchModal match={{ ...HIT, also_in: [{ ...REFERENCE, seq: 3 }] }} query="shadow" onClose={noop} />,
      contains: ['<details><summary class="section-row">', '<blockquote class="match-text also-text">Loading…</blockquote></details>'],
      missing: ['open=""'],
    },
    {
      name: 'the modal of a match with nothing folded shows no list',
      element: <MatchModal match={PASSAGE} query="shadow" onClose={noop} />,
      contains: [],
      missing: ['Also in'],
    },
  ])
})

describe('Statusbar', () => {
  const status = (models: Status['models']): Status => ({ initialized: true, home: '/home/ada/.haskie', embedding: null, models, settings_error: null })
  const bge = { kind: 'embedding' as const, name: 'BAAI/bge-small-en-v1.5', state: 'ready' as const, error: null, device: 'cpu' as const }
  const minilm = { kind: 'reranker' as const, name: 'Xenova/ms-marco-MiniLM-L-6-v2', state: 'loading' as const, error: null, device: 'cpu' as const }
  check([
    {
      name: 'a ready embedding is a green check, its name in the hint only',
      element: <Statusbar status={status([bge])} />,
      contains: [
        '<span class="muted">Embedding</span><span class="statusbar-counts"><b class="done" aria-label="ready">',
        '<span class="hint" role="tooltip"><span class="hint-rows"><span>BAAI/bge-small-en-v1.5</span><span class="muted"></span><span class="code">ready</span>',
      ],
      missing: ['Reranker', 'bge-small-en-v1.5 ·'],
    },
    {
      name: 'a reranker shows only when one is on, and a loading one spins',
      element: <Statusbar status={status([bge, minilm])} />,
      contains: ['<span class="muted">Reranker</span><span class="statusbar-counts"><b class="running" aria-label="loading">', '<span>Xenova/ms-marco-MiniLM-L-6-v2</span>'],
    },
    {
      name: 'no embedding model: full-text only, no hint',
      element: <Statusbar status={status([])} />,
      contains: ['<span class="muted">Embedding</span><span class="statusbar-counts"><b>full-text only</b></span></span>'],
      missing: ['role="tooltip"'],
    },
  ])
})

describe('SearchTook', () => {
  const steps = [
    { step: 'plan', label: 'Embed the query', ms: 12.34 },
    { step: 'retrieve', label: 'LanceDB retrieval', ms: 41.2 },
  ]
  check([
    {
      name: 'the total, and under it the server time, step by step',
      element: <SearchTook counts="3 excerpts" ms={80} steps={steps} />,
      contains: [
        '<p class="mono muted search-took" tabindex="0">3 excerpts · 80 ms<span class="hint hint-below" role="tooltip">',
        '<span class="label label-mono">Server · 54 ms</span>',
        '<span>LanceDB retrieval</span><span class="muted">retrieve</span><span class="code">41 ms</span>',
        '<span class="code">12 ms</span>',
      ],
    },
    {
      name: 'a search that timed no steps: the total alone, no hint',
      element: <SearchTook counts="3 chunks" ms={80} />,
      contains: ['<p class="mono muted search-took">3 chunks · 80 ms</p>'],
      missing: ['role="tooltip"'],
    },
    {
      name: 'before the first search: a blank line, no hint',
      element: <SearchTook counts="" ms={null} steps={steps} />,
      contains: ['<p class="mono muted search-took" tabindex="0"> </p>'],
      missing: ['role="tooltip"'],
    },
  ])
})
