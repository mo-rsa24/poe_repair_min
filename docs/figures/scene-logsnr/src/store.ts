/**
 * The one shared mathematical state. Every panel subscribes to this object;
 * no mathematical value lives anywhere else.
 *
 * log-SNR is deliberately absent as a stored field. It is derived from
 * (stepIndex, steps, spacing), and that derivation is the lesson.
 */
import { create } from 'zustand'
import { clampInt, type Spacing } from './math'
import { STORY } from './story'

export type Mode = 'story' | 'explore'

export type SceneState = {
  mode: Mode
  step: number
  playing: boolean

  alphaBar: number // warm-up panel: the signal's share of the power
  stepIndex: number // i, the step within sampler A's run
  steps: number // N for sampler A
  spacing: Spacing // spacing rule for sampler A
  compareSteps: number // N for sampler B
  compareSpacing: Spacing

  setMode: (m: Mode) => void
  goto: (step: number) => void
  next: () => void
  prev: () => void
  setPlaying: (p: boolean) => void
  patch: (p: Partial<SceneState>) => void
}

const reducedMotion = () =>
  typeof window !== 'undefined' &&
  !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** Ease one numeric field from its current value to a target. Direct
 *  manipulation bypasses this and writes state uneased. */
function ease(
  from: number,
  to: number,
  ms: number,
  apply: (v: number) => void,
  round = false,
) {
  if (reducedMotion() || from === to) {
    apply(to)
    return
  }
  const start = performance.now()
  const frame = (now: number) => {
    const p = Math.min(1, (now - start) / ms)
    const k = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2
    const v = from + (to - from) * k
    apply(round ? Math.round(v) : v)
    if (p < 1) requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

const EASED = new Set(['alphaBar', 'stepIndex'])

export const useScene = create<SceneState>((set, get) => ({
  mode: 'story',
  step: 0,
  playing: false,

  alphaBar: 0.5,
  stepIndex: 0,
  steps: 50,
  spacing: 'leading',
  compareSteps: 20,
  compareSpacing: 'trailing',

  setMode: (m) => set({ mode: m, playing: false }),

  goto: (step) => {
    const s = clampInt(step, 0, STORY.length - 1)
    const before = get()
    set({ step: s })
    for (const [key, value] of Object.entries(STORY[s].targets)) {
      if (!EASED.has(key)) {
        set({ [key]: value } as Partial<SceneState>)
        continue
      }
      const rounded = key === 'stepIndex'
      ease(
        before[key as keyof SceneState] as number,
        value as number,
        650,
        (v) => set({ [key]: v } as Partial<SceneState>),
        rounded,
      )
    }
  },

  next: () => get().goto(get().step + 1),
  prev: () => get().goto(get().step - 1),
  setPlaying: (p) => set({ playing: p }),
  patch: (p) => set(p),
}))
