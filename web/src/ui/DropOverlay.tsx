import { Upload } from 'lucide-react'
import { useEffect } from 'react'

/**
 * The whole window is a file target. Port of the `documents.html` script: `dragenter` and
 * `dragleave` fire for every child element, so a depth counter decides when the drag left the window.
 */
export function DropOverlay({ onFiles }: { onFiles: (files: File[]) => void }) {
  useEffect(() => {
    let depth = 0
    // a dragged link or piece of text is left to the browser: only files are taken
    const files = (event: DragEvent) => event.dataTransfer?.types.includes('Files') ?? false
    const enter = (event: DragEvent) => {
      if (!files(event)) return
      event.preventDefault()
      if (depth++ === 0) document.body.classList.add('over')
    }
    const over = (event: DragEvent) => {
      if (files(event)) event.preventDefault()
    }
    const leave = (event: DragEvent) => {
      if (files(event) && --depth === 0) document.body.classList.remove('over')
    }
    const drop = (event: DragEvent) => {
      if (!files(event)) return
      event.preventDefault()
      depth = 0
      document.body.classList.remove('over')
      onFiles([...(event.dataTransfer?.files ?? [])])
    }
    document.addEventListener('dragenter', enter)
    document.addEventListener('dragover', over)
    document.addEventListener('dragleave', leave)
    document.addEventListener('drop', drop)
    return () => {
      document.removeEventListener('dragenter', enter)
      document.removeEventListener('dragover', over)
      document.removeEventListener('dragleave', leave)
      document.removeEventListener('drop', drop)
      document.body.classList.remove('over')
    }
  }, [onFiles])

  return (
    <div className="drop-overlay" aria-hidden="true">
      <Upload className="icon" />
      Drop a file to add it
    </div>
  )
}
