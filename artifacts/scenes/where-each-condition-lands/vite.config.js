import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath } from 'node:url';
export default defineConfig({
  plugins: [svelte()],
  base: './',
  resolve: { alias: { '@diffusion-explorer/ui': fileURLToPath(new URL('./vendor/ui/index.ts', import.meta.url)) } },
});
