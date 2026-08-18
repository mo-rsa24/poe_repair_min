/**
 * Every mathematical function the scene uses. Nothing here is decorative:
 * each one is named in the scene spec and read by at least one panel.
 *
 * The schedule numbers are not invented. `data.json` carries SDXL's own
 * alphas_cumprod, read from the model repo by
 * poe_repair/experiments/interaction_term/cache.py, and the F3 curve is the
 * measured output of scripts/snr_collapse.py.
 */
import raw from './data.json'

export const ALPHA_BAR: number[] = raw.alphasCumprod
export const T_MAX = ALPHA_BAR.length // 1000 training timesteps
export const ALPHA_BAR_SOURCE: string = raw.alphasCumprodSource
export const F3 = raw.f3
export const F3_PATH: string = raw.f3Path

/** The 50 timesteps the cached runs actually visited, in sampler order
 *  (t descending, so log-SNR ascends). Not an assumed spacing rule: read from
 *  a cell's own meta.json. */
export const CACHED_TIMESTEPS: number[] = raw.cachedTimesteps
export const CACHED_TIMESTEPS_SOURCE: string = raw.cachedTimestepsSource

/** log-SNR at training timestep t. The definition used everywhere in the repo:
 *  lambda_t = log( alpha_bar_t / (1 - alpha_bar_t) ). */
export function lambdaAt(t: number): number {
  const ab = ALPHA_BAR[clampInt(t, 0, T_MAX - 1)]
  return Math.log(ab / (1 - ab))
}

/** The inverse direction, used by the round-trip check:
 *  alpha_bar = sigmoid(lambda). */
export function alphaBarFromLambda(lam: number): number {
  return 1 / (1 + Math.exp(-lam))
}

/** beta_t recovered from the cumulative product, for the schedule panel. */
export function betaAt(t: number): number {
  if (t === 0) return 1 - ALPHA_BAR[0]
  return 1 - ALPHA_BAR[t] / ALPHA_BAR[t - 1]
}

export const LAMBDA_MIN = lambdaAt(T_MAX - 1)
export const LAMBDA_MAX = lambdaAt(0)

/** Timestep t whose log-SNR is closest to lam. Binary search, since lambda is
 *  strictly decreasing in t (the monotone check proves it at runtime). */
export function timestepAtLambda(lam: number): number {
  let lo = 0
  let hi = T_MAX - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (lambdaAt(mid) > lam) lo = mid
    else hi = mid
  }
  return Math.abs(lambdaAt(lo) - lam) <= Math.abs(lambdaAt(hi) - lam) ? lo : hi
}

export type Spacing = 'leading' | 'trailing' | 'uniform-lambda'

export const SPACING_LABEL: Record<Spacing, string> = {
  leading: 'DDIM leading, evenly spaced in t',
  trailing: 'DDIM trailing, evenly spaced in t',
  'uniform-lambda': 'evenly spaced in log-SNR (idealised)',
}

/**
 * The N training timesteps one sampler actually visits, ordered the way the
 * sampler visits them: noisiest first, so log-SNR ascends with step index.
 *
 * `leading` and `trailing` are diffusers' two real spacing rules for DDIM.
 * `uniform-lambda` is an idealised schedule that puts its steps evenly along
 * the log-SNR axis instead of along t. It stands in for the EDM/Karras family
 * of samplers; it is illustrative of that family, not a run of this project.
 */
export function samplerTimesteps(n: number, spacing: Spacing): number[] {
  if (spacing === 'uniform-lambda') {
    const lo = lambdaAt(T_MAX - 1)
    const hi = lambdaAt(0)
    return Array.from({ length: n }, (_, i) =>
      timestepAtLambda(lo + ((hi - lo) * i) / (n - 1)),
    )
  }
  const ratio = Math.floor(T_MAX / n)
  const ts =
    spacing === 'leading'
      ? Array.from({ length: n }, (_, i) => i * ratio)
      : Array.from({ length: n }, (_, i) => T_MAX - i * ratio - 1)
  // descending t == ascending lambda == the order the sampler runs in
  return ts.sort((a, b) => b - a)
}

/* ---- the measured F3 curve, treated as one function of log-SNR ---- */

const G: number[] = F3.log_snr_grid
const M: number[] = F3.median_curve
export const F3_LO = G[0]
export const F3_HI = G[G.length - 1]

/** Linear interpolation of the measured median curve. Outside the measured
 *  log-SNR range the value is held flat at the end point; `inRange` says so,
 *  and every panel draws the held part differently rather than hiding it. */
export function correctionAt(lam: number): { y: number; inRange: boolean } {
  if (lam <= F3_LO) return { y: M[0], inRange: false }
  if (lam >= F3_HI) return { y: M[M.length - 1], inRange: false }
  let k = 0
  while (k < G.length - 2 && G[k + 1] < lam) k++
  const f = (lam - G[k]) / (G[k + 1] - G[k])
  return { y: M[k] + f * (M[k + 1] - M[k]), inRange: true }
}

export type Reading = {
  step: number // step index within this sampler's run
  u: number // step index rescaled to 0..1, the only way two runs of
  // different length can share an x axis at all
  t: number // training timestep visited
  lambda: number
  y: number // ||r_t|| / ||eps_PoE||, own-median scaled
  inRange: boolean
}

/** What one sampler reads off the same underlying correction curve. */
export function readings(n: number, spacing: Spacing): Reading[] {
  return samplerTimesteps(n, spacing).map((t, step) => {
    const lambda = lambdaAt(t)
    const { y, inRange } = correctionAt(lambda)
    return { step, u: n === 1 ? 0 : step / (n - 1), t, lambda, y, inRange }
  })
}

function interpBy<K extends 'u' | 'lambda'>(rs: Reading[], key: K, x: number): number | null {
  if (x < rs[0][key] || x > rs[rs.length - 1][key]) return null
  let k = 0
  while (k < rs.length - 2 && rs[k + 1][key] < x) k++
  const span = rs[k + 1][key] - rs[k][key]
  const f = span === 0 ? 0 : (x - rs[k][key]) / span
  return rs[k].y + f * (rs[k + 1].y - rs[k].y)
}

/** The scene's own claim, computed rather than asserted: the largest vertical
 *  gap between two samplers' curves, once on each candidate x axis. */
export function maxGap(a: Reading[], b: Reading[], key: 'u' | 'lambda'): number {
  let worst = 0
  for (const r of a) {
    const other = interpBy(b, key, r[key])
    if (other !== null) worst = Math.max(worst, Math.abs(r.y - other))
  }
  for (const r of b) {
    const other = interpBy(a, key, r[key])
    if (other !== null) worst = Math.max(worst, Math.abs(r.y - other))
  }
  return worst
}

/* ---- how the real run's steps land on each axis ---- */

export type Landing = { step: number; t: number; lambda: number }

/** The cached run, one entry per denoising step. */
export const CACHED: Landing[] = CACHED_TIMESTEPS.map((t, step) => ({
  step,
  t,
  lambda: lambdaAt(t),
}))

/** How far log-SNR moves at each step of the real run. Even in step index by
 *  construction; wildly uneven in log-SNR, which is the cost F3's step axis
 *  avoids. */
export function lambdaJumps(): number[] {
  return CACHED.slice(1).map((c, i) => c.lambda - CACHED[i].lambda)
}

/** How many real steps fall in each of `bins` evenly spaced log-SNR cells,
 *  spanning the run. This is the interpolation artefact, counted: a cell
 *  holding zero or one real step is a stretch of axis carrying no measurement
 *  of its own. */
export function stepsPerLambdaBin(bins: number): number[] {
  const lo = CACHED[0].lambda
  const hi = CACHED[CACHED.length - 1].lambda
  const centres = Array.from({ length: bins }, (_, i) => lo + ((hi - lo) * i) / (bins - 1))
  const half = (centres[1] - centres[0]) / 2
  return centres.map(
    (c, i) =>
      CACHED.filter(
        (x) =>
          x.lambda >= c - half - (i === 0 ? 1e-9 : 0) &&
          (x.lambda < c + half || (i === bins - 1 && x.lambda <= c + half + 1e-9)),
      ).length,
  )
}

/** The published F3 median curve, re-labelled onto the step axis by inverting
 *  lambda(t). Same twenty y values, same order; only the x positions change.
 *  This is a re-labelling of a published curve, not a recomputation from the
 *  per-cell data, which snr_collapse.py does not currently save. */
export function medianOnStepAxis(): [number, number][] {
  return F3.log_snr_grid.map((lam: number, i: number) => {
    // nearest real step, by log-SNR
    let best = 0
    for (let k = 1; k < CACHED.length; k += 1) {
      if (Math.abs(CACHED[k].lambda - lam) < Math.abs(CACHED[best].lambda - lam)) best = k
    }
    return [CACHED[best].step, F3.median_curve[i]] as [number, number]
  })
}

/* ---- live checks ---- */

/** Check 1: sigmoid(lambda) recovers the scheduler's own alpha_bar. */
export function roundTripError(): number {
  let worst = 0
  for (let t = 0; t < T_MAX; t += 1) {
    worst = Math.max(worst, Math.abs(alphaBarFromLambda(lambdaAt(t)) - ALPHA_BAR[t]))
  }
  return worst
}

/** Check 2: lambda strictly decreases in t, so nothing is lost by swapping
 *  the axis. Counts violations across all 1000 timesteps; must be 0. */
export function monotoneViolations(): number {
  let bad = 0
  for (let t = 1; t < T_MAX; t += 1) if (lambdaAt(t) >= lambdaAt(t - 1)) bad += 1
  return bad
}

/* ---- the first-year warm-up ---- */

/** A fixed 1-D "picture": one period of a smooth signal, so the noisy mixture
 *  can be drawn honestly rather than sketched. */
export const SIGNAL: number[] = Array.from({ length: 160 }, (_, i) => {
  const x = (i / 159) * Math.PI * 2
  return Math.sin(x) * 0.7 + Math.sin(2 * x + 1) * 0.3
})

/** One fixed noise draw, so moving the slider changes only the mixture and
 *  never re-rolls the randomness (the variable would be moving on its own). */
export const NOISE: number[] = (() => {
  let s = 12345
  return SIGNAL.map(() => {
    // deterministic Box-Muller off a small LCG, seeded once
    s = (1103515245 * s + 12345) % 2147483648
    const u1 = (s + 1) / 2147483649
    s = (1103515245 * s + 12345) % 2147483648
    const u2 = (s + 1) / 2147483649
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  })
})()

/** x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) eps, the whole forward process. */
export function noisySignal(alphaBar: number): number[] {
  const a = Math.sqrt(alphaBar)
  const b = Math.sqrt(1 - alphaBar)
  return SIGNAL.map((x0, i) => a * x0 + b * NOISE[i])
}

export function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)))
}

export function fmt(v: number, digits = 2): string {
  if (!isFinite(v)) return '—'
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0)) return v.toExponential(1)
  return v.toFixed(digits)
}
