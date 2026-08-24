// One shared state. Every panel reads it; no panel keeps a private copy.
import { create } from 'zustand'
import data from './data/result.json'
import type { ResultData, RowKey } from './types'

export const D = data as unknown as ResultData

export const ALL_PAIRS = [...new Set(D.cells.map((c) => c.pair))].sort()
export const ALL_SEEDS = [...new Set(D.cells.map((c) => c.seed))].sort((a, b) => a - b)

type State = {
  claim: string
  lam: number
  row: RowKey
  pair: string
  seed: number
  pairs: string[]
  seeds: number[]
  lightbox: string | null
  set: (p: Partial<State>) => void
  togglePair: (p: string) => void
  toggleSeed: (s: number) => void
  resetSelection: () => void
}

export const useScene = create<State>((set) => ({
  claim: 'C1',
  lam: 1,
  row: 'oracle',
  // The strip pair and seed the review file argues from, so the scene opens where the
  // paper's figure does.
  pair: 'a_cat__x__a_dog',
  seed: 9,
  pairs: ALL_PAIRS,
  seeds: ALL_SEEDS,
  lightbox: null,
  set: (p) => set(p),
  togglePair: (p) =>
    set((s) => ({ pairs: s.pairs.includes(p) ? s.pairs.filter((x) => x !== p) : [...s.pairs, p] })),
  toggleSeed: (n) =>
    set((s) => ({ seeds: s.seeds.includes(n) ? s.seeds.filter((x) => x !== n) : [...s.seeds, n] })),
  resetSelection: () => set({ pairs: ALL_PAIRS, seeds: ALL_SEEDS }),
}))

export const ROW_COLOR: Record<RowKey, string> = {
  oracle: 'var(--oracle)',
  random: 'var(--control-a)',
  wrong_pair: 'var(--control-b)',
}
