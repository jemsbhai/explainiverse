import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    root: 'src/demo',
    base: process.env.GITHUB_PAGES ? '/explainiverse/' : '/',
    build: {
        outDir: '../../dist/demo',
        emptyOutDir: true,
    },
});
