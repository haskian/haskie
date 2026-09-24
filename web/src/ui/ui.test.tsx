import { describe, expect, test } from 'bun:test'
import { FileText } from 'lucide-react'
import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { Hit, Passage, Source } from '../api'
import { Field } from './Field'
import { GallerySection } from './GallerySection'
import { HitGrid } from './HitGrid'
import { Kv } from './Kv'
import { MatchModal } from './MatchModal'
import { Mark } from './Mark'
import { markTerms } from './markTerms'
import { Picker } from './Picker'
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
        '<span>Soft shadows</span>',
        'p. 2 · ',
        'chunk ',
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
      name: 'a passage names its chunk run and the last step of its breadcrumb',
      element: <HitGrid results={[PASSAGE]} query="shadow" />,
      contains: ['<span>Soft shadows</span>', '<span>p. 2 · chunks 4–5</span>'],
    },
    {
      name: 'a passage of one chunk names that chunk by its sequence number',
      element: <HitGrid results={[{ ...PASSAGE, seq_end: 4, page_start: null }]} query="shadow" />,
      contains: ['<span>chunk 4 · lines 41–58</span>'],
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
      name: 'a chunk of headings alone shows them grey, under the path above them',
      element: (
        <MatchModal
          match={{ ...HIT, headings: ['Book', 'Index', 'Terms'], frame: ['Book'], text: '# Index\n\n## Terms', layout: [{ type: 'heading', position: 0 }, { type: 'heading', position: 9 }] }}
          query=""
          onClose={noop}
        />
      ),
      contains: [
        '<span class="chunk-piece chunk-frame">Book\n\n',
        '<span class="chunk-piece chunk-heading"># Index\n\n<span class="hint" role="tooltip"><strong>Heading</strong><span class="sub mono">position 0 · 9 chars · 1 words</span></span></span>',
        '<span class="chunk-piece chunk-heading">## Terms<span class="hint" role="tooltip"><strong>Heading</strong>',
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
