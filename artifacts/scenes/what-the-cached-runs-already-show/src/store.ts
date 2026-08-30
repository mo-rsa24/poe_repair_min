// One shared state. Every panel reads it; no panel keeps a private copy.
// Move the step and the curve marker, the readout and the pictures all move,
// because they are one object.
import { create } from 'zustand'
import data from './data/result.json'
import type { ExpertCell, ExpertRow, ResultData } from './types'

export const D = data as unknown as ResultData

export const EXPERT_PAIRS = [...new Set(D.experts.cells.map((c) => c.pair))].sort()
export const FORK_PAIRS = [...new Set(D.fork.refreshed.cells.map((c) => c.pair))].sort()

type State = {
  claim: string
  /** index into the collapse log-SNR grid, 0..19 */
  bin: number
  /** sampling step, 0..50 */
  step: number
  /** which pair and seed the pictures show */
  pair: string
  seed: number
  /** the fork cell under inspection */
  forkPair: string
  forkSeed: number
  /** rank cut for the spectrum */
  k: number
  /** which size measure C2 is showing */
  measure: 'prereg' | 'raw' | 'both'
  /** whether the fork read covers 43 cells or the review's original 19 */
  forkCoverage: 'refreshed' | 'original'
  lightbox: string | null
  set: (p: Partial<State>) => void
}

const firstExpert = D.experts.cells[0]

/** The claim lives in the URL hash, so a panel can be linked to and reloaded. */
const VALID = new Set([...D.claims.map((c) => c.id), 'owed'])
const fromHash = () => {
  const h = window.location.hash.replace('#', '')
  return VALID.has(h) ? h : 'C1'
}

export const useScene = create<State>((set) => ({
  claim: fromHash(),
  bin: 10,
  step: 16, // the fork elbow, so the scene opens where the result is
  pair: firstExpert.pair,
  seed: firstExpert.seed,
  forkPair: 'a_cat__x__a_dog',
  forkSeed: 1,
  k: 8, // the rank the LoRA actually uses
  measure: 'both',
  forkCoverage: 'refreshed',
  lightbox: null,
  set: (p) => {
    if (p.claim) window.location.hash = p.claim
    set(p)
  },
}))

window.addEventListener('hashchange', () => {
  useScene.setState({ claim: fromHash() })
})

/** The expert-frame cell currently selected, or null when that pair has none. */
export function expertCell(pair: string, seed: number) {
  return (
    D.experts.cells.find((c) => c.pair === pair && c.seed === seed) ??
    D.experts.cells.find((c) => c.pair === pair) ??
    null
  )
}

/** The decoded row nearest a given sampling step. Frames exist every 5 steps. */
export function nearestRow(cell: ExpertCell, step: number): ExpertRow {
  return cell.rows.reduce((best, r) =>
    Math.abs(r.step - step) < Math.abs(best.step - step) ? r : best,
  )
}

/** The decoded row nearest a log-SNR value, so C1's scrubber can drive pictures. */
export function rowNearestLogSnr(cell: ExpertCell, logSnr: number): ExpertRow {
  return cell.rows.reduce((best, r) =>
    Math.abs(r.logSnr - logSnr) < Math.abs(best.logSnr - logSnr) ? r : best,
  )
}

export const COLOR = {
  poe: 'var(--poe)',
  mono: 'var(--mono)',
  lora: 'var(--lora)',
  controlA: 'var(--control-a)',
  controlB: 'var(--control-b)',
  floor: 'var(--floor)',
}
