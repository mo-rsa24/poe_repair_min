// The shape the loader emits. Nothing here holds a value; values live in
// src/data/result.json and every one of them carries the file it came from.

export type Stamp = { path: string; mtime: string; bytes: number }
export type Quote = { path: string; lines: [number, number]; body: string[] }

export type CellRef = { pair: string; seed: number }

export type Curve = {
  normalize: string
  grid: number[]
  median: number[]
  iqr: number[]
  spreadPct: number
  verdict: string
  peakLogSnr: number
  peakAtEdge: boolean
  nPairs: number
  nCurves: number
  cells: CellRef[]
}

export type ForkCell = CellRef & { elbowStep: number; distance: number[] }
export type ForkRead = {
  medianElbow: number
  nCells: number
  elbowMin: number
  elbowMax: number
  inBand13to20: number
  maxDistanceAtZero: number
  cells: ForkCell[]
}

export type ClimbCell = CellRef & {
  normalised: number
  raw: number
  controlRandom: number
  controlWrongStep: number
  rVsEpsPoe: number
  epsVsDx: number
  fractionNegative: number
  perStepCosine: number[]
}

export type ExpertRow = {
  step: number
  timestep: number
  logSnr: number
  views: Record<string, string>
}

export type ExpertCell = CellRef & {
  promptA: string
  promptB: string
  nSteps: number
  px: number
  rows: ExpertRow[]
}

export type Claim = {
  id: string
  type: string
  mark: string | null
  state: 'measured' | 'not-run'
  question: string
  quote: Quote
  reads: string[]
  knob?: string
}

export type Gap = {
  id: string
  closes: string
  what: string
  why: string
  command: string
  output: string
  cost: string
}

export type Discrepancy = {
  claim: string
  says: string
  found: string
  effect: string
}

export type ResultData = {
  meta: {
    builtFrom: Record<string, Stamp>
    commit: string
    roots: Record<string, string>
    publicLinks: Record<string, string>
  }
  verdict: Quote
  runKind: Quote
  vocabulary: Quote[]
  openQuestion: Quote
  claims: Claim[]
  collapse: {
    committed: Curve
    prereg: Curve
    raw: Curve
    perCellCurvesOnDisk: boolean
  }
  fork: { original: ForkRead; refreshed: ForkRead }
  forkImages: (CellRef & { poe: string | null; mono: string | null })[]
  climb: {
    measure: string
    caveat: string
    reading: string
    nCells: number
    nNegative: number
    nPairs: number
    medians: Record<string, number>
    cells: ClimbCell[]
    otherPopulation: {
      nCells: number
      nPairs: number
      climbMedian: number
      alignmentMedian: number
      randomFloor: number
    }
  }
  spectrum: {
    ks: number[]
    energy: number[]
    floor: number[]
    heldout: number[]
    singularValues: number[]
    trainPairs: string[]
    heldoutPairs: string[]
    trainVectors: number
    dims: number
    transfer: {
      perPair: { pair: string; composeRate: number; geometryK64: number }[]
      meanCompose: number
      meanGeometry: number
      evalStep: string
      heldoutVectors: number
    }
  }
  experts: { views: string[]; cells: ExpertCell[] }
  gaps: Gap[]
  discrepancies: Discrepancy[]
}
