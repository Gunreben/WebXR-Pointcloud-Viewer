import { cpSync, existsSync } from 'node:fs';
import path from 'node:path';
import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

function copyPointclouds() {
  return {
    name: 'copy-pointclouds',
    apply: 'build',
    writeBundle() {
      const sourceDir = path.resolve(__dirname, 'Pointclouds');
      const outputDir = path.resolve(__dirname, 'dist', 'Pointclouds');

      if (existsSync(sourceDir)) {
        cpSync(sourceDir, outputDir, { recursive: true });
      }
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [basicSsl(), copyPointclouds()],
  server: {
    https: true,
    host: true,
  },
});
