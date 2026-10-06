import { createContext } from 'react'

// Undefined outside a modal; null until its footer mounts. Nested dialogs own separate targets.
export const ModalStatusTarget = createContext<HTMLDivElement | null | undefined>(undefined)
