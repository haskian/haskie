import { X } from 'lucide-react'
import { useLayoutEffect, useRef, type ReactNode } from 'react'

/**
 * Native `<dialog>`: `showModal()` gives the backdrop, the focus trap and Escape for free. The
 * children mount only while it is open, so a panel that fetches does not fetch behind a closed
 * modal.
 */
export function Modal({
  open,
  onClose,
  title,
  subtitle,
  children,
}: {
  open: boolean
  onClose: () => void
  title: string
  subtitle?: string
  children: ReactNode
}) {
  const dialog = useRef<HTMLDialogElement>(null)

  // Layout, not passive: the dialog is open before the browser paints or loads anything in it. A
  // PDF viewer that starts inside a closed (display: none) dialog stays blank in Chrome.
  useLayoutEffect(() => {
    const element = dialog.current
    if (!element) return
    if (open && !element.open) element.showModal()
    if (!open && element.open) element.close()
  }, [open])

  return (
    <dialog
      className="modal"
      ref={dialog}
      onClose={(event) => {
        // React bubbles `close` through the component tree, unlike the DOM: a nested modal closing
        // must not close this one too.
        if (event.target !== dialog.current) return
        if (open) onClose() // Escape and the close button end the dialog without telling the caller
      }}
      onClick={(event) => {
        if (event.target === dialog.current) onClose() // the backdrop is the dialog itself; .modal-body is the content
      }}
    >
      {open && (
        <div className="modal-body" tabIndex={-1} autoFocus>
          <div className="modal-head">
            <h2 className="title">
              <span>{title}</span>
              {subtitle !== undefined && <small className="muted">{subtitle}</small>}
            </h2>
            <form method="dialog">
              <button className="btn btn-ghost" aria-label="Close">
                <X className="icon" />
              </button>
            </form>
          </div>
          {children}
        </div>
      )}
    </dialog>
  )
}
