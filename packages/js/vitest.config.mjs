/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    test: {
        environment: 'jsdom',
        globals: true, // needed for jest-dom matchers if extended globally?
        setupFiles: ['./tests/setup.ts'],
    },
});
