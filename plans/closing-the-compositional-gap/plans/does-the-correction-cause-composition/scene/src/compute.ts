// Every number the page shows is computed here from the loaded cells, then compared
// against the values the results file carries. Nothing is transcribed.
import type { Cell, ResultData, RowKey } from './types'

export type Selection = { pairs: string[]; seeds: number[] }

export function selectCells(cells: Cell[], sel: Selection): Cell[] {
  return cells.filter((c) => sel.pairs.includes(c.pair) && sel.seeds.includes(c.seed))
}

/** Compose rate for one row at one strength, over whatever cells are selected. */
export function rate(cells: Cell[], row: RowKey, lam: number) {
  const hit = cells.filter((c) => c.row === row && c.lam === lam)
  const n = hit.length
  const k = hit.reduce((a, c) => a + c.compose, 0)
  return { rate: n ? k / n : NaN, n, k }
}

export function curve(cells: Cell[], row: RowKey, lambdas: number[]) {
  return lambdas.map((l) => rate(cells, row, l))
}

/** Trapezoid over the strength axis. This is the rule the results file used: its own
 *  AUC values reproduce exactly under it, which the page shows rather than assumes. */
export function auc(points: number[], lambdas: number[]) {
  let a = 0
  for (let i = 1; i < points.length; i++) {
    a += ((points[i] + points[i - 1]) / 2) * (lambdas[i] - lambdas[i - 1])
  }
  return a
}

export type RowSummary = {
  row: RowKey
  points: { rate: number; n: number; k: number }[]
  auc: number
  fileAuc: number
  fileCurve: number[]
  agreesWithFile: boolean
  cellsPerPoint: number[]
  equalCells: boolean
  rise: number
}

export function summarize(data: ResultData, sel: Selection): RowSummary[] {
  const cells = selectCells(data.cells, sel)
  return data.rows.map(({ key }) => {
    const points = curve(cells, key, data.lambdas)
    const rates = points.map((p) => p.rate)
    const a = auc(rates, data.lambdas)
    const fileCurve = data.fileSummary.curves[key]
    const fileAuc = data.fileSummary.auc[key]
    const cellsPerPoint = points.map((p) => p.n)
    return {
      row: key,
      points,
      auc: a,
      fileAuc,
      fileCurve,
      // Only meaningful when the whole sweep is selected; the page says so.
      agreesWithFile:
        rates.every((r, i) => Math.abs(r - fileCurve[i]) < 1e-9) && Math.abs(a - fileAuc) < 1e-9,
      cellsPerPoint,
      equalCells: new Set(cellsPerPoint).size === 1,
      rise: rates[rates.length - 1] - rates[0],
    }
  })
}

/** Does this seed's oracle row rise at every step? The review says seed 10 does and
 *  seeds 9 and 11 do not; the page recomputes it rather than repeating it. */
export function monotoneBySeed(data: ResultData, pairs: string[]) {
  const seeds = [...new Set(data.cells.map((c) => c.seed))].sort((a, b) => a - b)
  return seeds.map((seed) => {
    const cells = selectCells(data.cells, { pairs, seeds: [seed] })
    const pts = curve(cells, 'oracle', data.lambdas).map((p) => p.rate)
    let monotone = true
    for (let i = 1; i < pts.length; i++) if (pts[i] < pts[i - 1]) monotone = false
    return { seed, points: pts, monotone }
  })
}

/** One cell, for putting a picture on a mark. */
export function oneCell(
  data: ResultData,
  sel: { row: RowKey; pair: string; seed: number; lam: number },
) {
  return data.cells.find(
    (c) => c.row === sel.row && c.pair === sel.pair && c.seed === sel.seed && c.lam === sel.lam,
  )
}

export function cellsFor(data: ResultData, sel: { pair: string; seed: number; lam: number }) {
  return data.rows.map(({ key }) => ({
    row: key,
    cell: data.cells.find(
      (c) => c.row === key && c.pair === sel.pair && c.seed === sel.seed && c.lam === sel.lam,
    ),
  }))
}

export const pct = (x: number) => (Number.isNaN(x) ? 'no cells' : `${Math.round(x * 100)}%`)
export const pct1 = (x: number) => (Number.isNaN(x) ? 'no cells' : `${(x * 100).toFixed(1)}%`)
export const fmt = (x: number, d = 3) => x.toFixed(d)
export const prettyPair = (p: string) => p.replace(/__x__/g, ' × ').replace(/_/g, ' ')
export const thumbUrl = (relPath: string) => `/thumbs/${relPath.replace(/\.png$/, '.jpg')}`
export const fullUrl = (relPath: string) => `/full/${relPath}`
