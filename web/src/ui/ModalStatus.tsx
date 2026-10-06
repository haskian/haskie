import { CircleAlert, CircleCheck, Info, TriangleAlert } from 'lucide-react'
import { useContext, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ModalStatusTarget } from './modalStatusTarget'

const ICONS = { info: Info, success: CircleCheck, warning: TriangleAlert, error: CircleAlert }

/** Messages follow their owning dialog, even when raised deep inside a panel. */
export function ModalStatus({ tone = 'info', children }: {
  tone?: keyof typeof ICONS
  children: ReactNode
}) {
  const target = useContext(ModalStatusTarget)
  const Icon = ICONS[tone]
  const message = (
    <div className={`modal-message modal-message-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon className="icon" aria-hidden="true" />
      <span>{children}</span>
    </div>
  )
  if (target === undefined) return message
  return target === null ? null : createPortal(message, target)
}
