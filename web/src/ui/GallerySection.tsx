import type { ReactNode } from 'react'

/** One labelled band of tiles. `large` swaps in the wider `gallery-lg` grid. */
export function GallerySection({
  label,
  id,
  large,
  children,
}: {
  label: string
  id?: string
  large?: boolean
  children: ReactNode
}) {
  return (
    <section className="gallery-section section" id={id}>
      <span className="mono muted">{label}</span>
      <div className={large ? 'gallery gallery-lg' : 'gallery'}>{children}</div>
    </section>
  )
}
