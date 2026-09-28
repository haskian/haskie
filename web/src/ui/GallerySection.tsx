import type { ReactNode } from 'react'

/** One labelled band of tiles. `large` swaps in the wider `gallery-lg` grid; `collapsed` folds
 *  the band behind its label, a native `<details>` the reader opens. */
export function GallerySection({
  label,
  id,
  large,
  collapsed,
  children,
}: {
  label: string
  id?: string
  large?: boolean
  collapsed?: boolean
  children: ReactNode
}) {
  const gallery = <div className={large ? 'gallery gallery-lg' : 'gallery'}>{children}</div>
  if (collapsed) {
    return (
      <details className="gallery-section section" id={id}>
        <summary className="mono muted">{label}</summary>
        {gallery}
      </details>
    )
  }
  return (
    <section className="gallery-section section" id={id}>
      <span className="mono muted">{label}</span>
      {gallery}
    </section>
  )
}
