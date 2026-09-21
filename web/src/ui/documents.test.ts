import { describe, expect, test } from 'bun:test'
import { File, FileCode, FileText, Image as ImageIcon } from 'lucide-react'
import { documentIcon, groupByRange, nameRange } from './documents'

describe('documentIcon', () => {
  const cases: Array<{ name: string; value: string; expected: ReturnType<typeof documentIcon> }> = [
    { name: 'pdf', value: '.pdf', expected: FileText },
    { name: 'image', value: '.png', expected: ImageIcon },
    { name: 'another image suffix', value: '.jpeg', expected: ImageIcon },
    { name: 'markdown', value: '.md', expected: FileCode },
    { name: 'plain text', value: '.txt', expected: FileCode },
    { name: 'anything else', value: '.docx', expected: File },
    { name: 'no suffix at all', value: '', expected: File },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(documentIcon(one.value)).toBe(one.expected)
    })
  }
})

describe('nameRange', () => {
  const cases: Array<{ name: string; value: string; expected: string }> = [
    { name: 'the first letter of the first band', value: 'Area', expected: 'A–E' },
    { name: 'the last letter of the first band', value: 'Eye', expected: 'A–E' },
    { name: 'the letter after it starts the next band', value: 'Film', expected: 'F–J' },
    { name: 'the middle band', value: 'Lamp', expected: 'K–O' },
    { name: 'the fourth band', value: 'Peak', expected: 'P–T' },
    { name: 'the last band', value: 'Wind', expected: 'U–Z' },
    { name: 'Z is absorbed by the last band', value: 'Zap', expected: 'U–Z' },
    { name: 'a lower-case name bands by its upper-case letter', value: 'eye.pdf', expected: 'A–E' },
    { name: 'leading space is ignored', value: '  Box', expected: 'A–E' },
    { name: 'a digit is not a letter', value: '2019-report.pdf', expected: '#' },
    { name: 'a letter outside A–Z is not a letter either', value: 'Émile', expected: '#' },
    { name: 'an empty name has no first letter', value: '', expected: '#' },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(nameRange(one.value)).toBe(one.expected)
    })
  }
})

// The galleries band whatever carries a name, so the fixture is the name-carrying row itself.
interface Named {
  name: string
}
const named = (name: string): Named => ({ name })

describe('groupByRange', () => {
  const cases: Array<{ name: string; value: Named[]; expected: Array<[string, string[]]> }> = [
    { name: 'nothing to band', value: [], expected: [] },
    { name: 'one band with one row', value: [named('Area')], expected: [['A–E', ['Area']]] },
    {
      name: 'bands follow the alphabet, not the rows, and empty ones are left out',
      value: [named('Wind'), named('Area'), named('Box')],
      expected: [
        ['A–E', ['Area', 'Box']],
        ['U–Z', ['Wind']],
      ],
    },
    {
      name: 'names that start with no letter band last',
      value: [named('2019-report.pdf'), named('Area')],
      expected: [
        ['A–E', ['Area']],
        ['#', ['2019-report.pdf']],
      ],
    },
    {
      name: 'rows inside a band keep the listing order',
      value: [named('Cone'), named('Area')],
      expected: [['A–E', ['Cone', 'Area']]],
    },
  ]
  for (const one of cases) {
    test(one.name, () => {
      expect(groupByRange(one.value, (item) => item.name).map((group) => [group.label, group.items.map((item) => item.name)])).toEqual(
        one.expected,
      )
    })
  }
})
