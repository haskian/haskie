// The background choice is one preference with two halves (what is stored, and the body
// attribute the stylesheet keys on), so both live here: `Settings` writes it, `App` applies what
// a previous visit stored.
const BACKGROUND_KEY = 'haskie.bg'
const CLASSIC = 'classic'

/** Whether the classic background is the stored choice. */
export const classicBackground = (): boolean => localStorage.getItem(BACKGROUND_KEY) === CLASSIC

/** Put the stored background choice, if any, on the body. */
export function applyBackground(): void {
  const stored = localStorage.getItem(BACKGROUND_KEY)
  if (stored) document.body.dataset.bg = stored
  else delete document.body.dataset.bg
}

/** Store the choice and apply it at once, so the page reacts as the toggle is clicked. */
export function setBackground(classic: boolean): void {
  if (classic) localStorage.setItem(BACKGROUND_KEY, CLASSIC)
  else localStorage.removeItem(BACKGROUND_KEY)
  applyBackground()
}
