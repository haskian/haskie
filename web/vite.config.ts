import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
    // src/index.css and src/ui/Shell.tsx read design/, which sits above the Vite root.
    fs: { allow: ['..'] },
  },
})
