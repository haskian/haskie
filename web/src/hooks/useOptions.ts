import { use } from 'react'
import { api, type Options } from '../api'

// The option catalogue, read straight from the one memoised request. `use` suspends the first
// render until it arrives, so every view gets the catalogue as a plain value instead of a
// `null`-until-loaded state. That includes the hooks that need a status list to answer at all.
export function useOptions(): Options {
  return use(api.options())
}
