// Formats shared by the views, built once: a formatter is expensive to create and the listings
// format a value per row.
export const bytes = new Intl.NumberFormat(undefined, {
  notation: 'compact',
  style: 'unit',
  unit: 'byte',
  unitDisplay: 'narrow',
})

// One place knows the API speaks epoch seconds.
export const at = (seconds: number) => new Date(seconds * 1000)
// Date and time of a timestamp; 0 is "never happened" and shows as nothing.
export const when = (seconds: number) => (seconds ? at(seconds).toLocaleString() : '')
