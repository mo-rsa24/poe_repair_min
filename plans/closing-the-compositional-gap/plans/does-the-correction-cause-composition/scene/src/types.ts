export type Src = { path: string; line?: number; lines?: [number, number] }
export type Stamp = { path: string; abs: string; mtime: string; bytes: number }

export type Cell = {
  row: RowKey
  pair: string
  seed: number
  lam: number
  nInstances: number
  compose: number
  image: { relPath: string; exists: boolean; sharedAtZero: boolean }
}

export type RowKey = 'oracle' | 'random' | 'wrong_pair'

export type Claim = {
  id: string
  type: string
  state: 'measured' | 'quoted'
  mark: string | null
  question: string
  answer: string[]
  source: { path: string; lines: [number, number] }
  fill: {
    what: string
    command: string
    output: string
    cost: string
    hasCommand: boolean
  } | null
}

export type ResultData = {
  meta: {
    builtFrom: Record<string, Stamp>
    imagesRoot: string
    figuresRoot: string
    runIdNote: string
    runLog: string
    missingImages: number
    thumbs?: { dir: string; px: number; written: number; upToDate: number; unique: number }
    publicLinks: Record<string, string>
  }
  verdict: { path: string; lines: [number, number]; body: string[] }
  runKind: Src & { text: string }
  constants: Record<'minBoxFraction' | 'sweepSeeds', { name: string; raw: string; path: string; line: number }>
  rows: { key: RowKey; label: string; isControl: boolean }[]
  lambdas: number[]
  cells: Cell[]
  fileSummary: {
    curves: Record<RowKey, number[]>
    auc: Record<RowKey, number>
    nCells: Record<RowKey, number>
    scorer: string
    note: string
  }
  claims: Claim[]
  supersededTable: { path: string; lines: [number, number]; rows: string[][]; why: string }
  environment: {
    outputBytes: number
    outputFilesystem: string
    scriptWrites: Src & { text: string }
    scriptGuardChecks: Src & { text: string }
    guardFilesystem: string
    planTask: Src & { text: string }
  }
  figures: Record<string, (Stamp & { publicPath: string }) | null>
}
