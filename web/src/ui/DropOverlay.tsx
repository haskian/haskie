import { Upload } from 'lucide-react'
import { useEffect } from 'react'

/**
 * The whole window is a file target. Port of the `documents.html` script: `dragenter` and
 * `dragleave` fire for every child element, so a depth counter decides when the drag really left.
 */
export function DropOverlay({ onFiles }: { onFiles: (files: File[]) => void }) {
  useEffect(() => {
    let depth = 0
    const enter = (event: DragEvent) => {
      event.preventDefault()
      if (depth++ === 0) document.body.classList.add('over')
    }
    const over = (event: DragEvent) => event.preventDefault()
    const leave = () => {
      if (--depth === 0) document.body.classList.remove('over')
    }
    const drop = (event: DragEvent) => {
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
      Drop files to upload
    </div>
  )
}
