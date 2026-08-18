import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The pictures are not copied into the app: public/experts and public/forkcells
// are symlinks to where they actually live, so both roots must be readable.
// When an output moves, rerun the loader and add the new root here.
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: [
        '.',
        '/datasets/mmolefe/poe_repair_min/outputs/interaction_term/experts',
        '/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/outputs/interaction_term/dose/pairs',
      ],
    },
  },
})
