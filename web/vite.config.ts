import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {
    // strictPort: a taken port is an error, not a silent move to the next one.
    port: 8453,
    strictPort: true,
    proxy: { '/api': `http://127.0.0.1:${process.env.HASKIE_PORT ?? '8451'}` },
    // src/index.css and src/ui/Shell.tsx read design/, which sits above the Vite root.
    fs: { allow: ['..'] },
  },
})
