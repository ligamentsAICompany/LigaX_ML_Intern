import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        ws: true, // Proxy WebSocket connections (/api/ws/...)
      },
      '/auth': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks(id) {
          const normalizedId = id.replace(/\\/g, '/');
          if (!normalizedId.includes('node_modules')) return;
          if (normalizedId.includes('@mui') || normalizedId.includes('@emotion')) return 'mui';
          if (
            normalizedId.includes('react-syntax-highlighter') ||
            normalizedId.includes('refractor') ||
            normalizedId.includes('prismjs')
          ) {
            return 'syntax-highlighter';
          }
          if (
            normalizedId.includes('react-markdown') ||
            normalizedId.includes('remark-') ||
            normalizedId.includes('micromark') ||
            normalizedId.includes('unified') ||
            normalizedId.includes('hast') ||
            normalizedId.includes('mdast') ||
            normalizedId.includes('unist') ||
            normalizedId.includes('vfile') ||
            normalizedId.includes('bail') ||
            normalizedId.includes('ccount') ||
            normalizedId.includes('character-entities') ||
            normalizedId.includes('comma-separated-tokens') ||
            normalizedId.includes('decode-named-character-reference') ||
            normalizedId.includes('devlop') ||
            normalizedId.includes('html-url-attributes') ||
            normalizedId.includes('longest-streak') ||
            normalizedId.includes('property-information') ||
            normalizedId.includes('space-separated-tokens') ||
            normalizedId.includes('stringify-entities') ||
            normalizedId.includes('trim-lines') ||
            normalizedId.includes('zwitch')
          ) {
            return 'markdown';
          }
          if (normalizedId.includes('@ai-sdk') || normalizedId.includes('/ai/')) return 'ai-sdk';
        },
      },
    },
  },
})
