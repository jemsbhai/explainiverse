/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    test: {
        environment: 'happy-dom',
        exclude: ['tests/browser/**', 'node_modules/**', 'dist/**'],
        globals: true,
        setupFiles: ['./tests/setup.ts'],
    },
});
