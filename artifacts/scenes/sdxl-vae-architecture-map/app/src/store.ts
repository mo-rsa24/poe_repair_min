import { create } from "zustand";
import { SYNTH } from "./data";

/** The one shared state: current node, stepper clock, hover token, real activations. */
interface SceneState {
  path: string;
  step: number; // the one clock: conv stepper window position, attention query index
  hoverToken: string | null; // visual element under the pointer; CodePane highlights matching lines
  activations: Record<string, number[]> | null; // vae_act.json when exported, else null
  navigate: (path: string) => void;
  setStep: (n: number) => void;
  setHover: (t: string | null) => void;
  setActivations: (a: Record<string, number[]>) => void;
}

function pathFromHash(): string {
  const h = window.location.hash.replace(/^#\/?/, "");
  return decodeURIComponent(h);
}

export const useScene = create<SceneState>((set) => ({
  path: pathFromHash(),
  step: 0,
  hoverToken: null,
  activations: null,
  navigate: (path) => {
    window.location.hash = "/" + encodeURIComponent(path);
    set({ path, step: 0, hoverToken: null });
  },
  setStep: (n) => set({ step: n }),
  setHover: (t) => set({ hoverToken: t }),
  setActivations: (a) => set({ activations: a }),
}));

window.addEventListener("hashchange", () => {
  useScene.setState({ path: pathFromHash(), step: 0, hoverToken: null });
});

export { SYNTH };
