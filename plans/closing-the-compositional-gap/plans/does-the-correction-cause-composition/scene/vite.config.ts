import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The cells are 3.4GB and are not copied into the app: public/full and public/figures
// are symlinks to where they actually live, so both roots must be readable. When the
// output moves off /home-mscluster, rerun the loader and add the new root here.
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: [
        '.',
        '/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/outputs/interaction_term/dose',
        '/datasets/mmolefe/poe_repair_min/outputs/interaction_term/dose',
        '/home-mscluster/mmolefe/Playground/PhD/poe_repair_min/paper/iclr/figures',
      ],
    },
  },
})
