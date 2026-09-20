import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './App.css'

// The boundary `useOptions` suspends on: the option catalogue is one request for the whole app,
// so one fallback at the root is the whole of its loading state.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Suspense fallback={<p className="muted">loading…</p>}>
      <App />
    </Suspense>
  </StrictMode>,
)
