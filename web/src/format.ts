// Number formats shared by the views, built once: a formatter is expensive to create and the
// listings format a value per row.
export const bytes = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  style: 'unit',
  unit: 'byte',
  unitDisplay: 'narrow',
})
