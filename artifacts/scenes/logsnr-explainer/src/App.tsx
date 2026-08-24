import { useEffect, useMemo } from 'react'
import { motion } from 'motion/react'
import { Plot } from './Plot'
import { Tex } from './Tex'
import { Details } from './Details'
import { useScene } from './store'
import { STORY } from './story'
import {
  ALPHA_BAR,
  ALPHA_BAR_SOURCE,
  CACHED,
  CACHED_TIMESTEPS_SOURCE,
  F3,
  F3_HI,
  F3_LO,
  F3_PATH,
  SPACING_LABEL,
  T_MAX,
  alphaBarFromLambda,
  betaAt,
  fmt,
  lambdaAt,
  lambdaJumps,
  maxGap,
  medianOnStepAxis,
  monotoneViolations,
  noisySignal,
  readings,
  roundTripError,
  samplerTimesteps,
  stepsPerLambdaBin,
  type Spacing,
} from './math'

const LAM = 'var(--lam)'
const IDX = 'var(--idx)'
const INK = 'var(--ink-soft)'
const TS = Array.from({ length: 201 }, (_, i) => Math.min(T_MAX - 1, i * 5))
const BINS = 20

export default function App() {
  const s = useScene()
  const lit = useMemo(() => new Set(STORY[s.step].lit), [s.step])
  const on = (name: string) => (s.mode === 'explore' || lit.has(name) ? 1 : 0.15)

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (s.mode !== 'story') return
      if (e.key === 'ArrowRight') s.next()
      if (e.key === 'ArrowLeft') s.prev()
      if (e.key === ' ') {
        e.preventDefault()
        s.setPlaying(!s.playing)
      }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  })

  useEffect(() => {
    if (!s.playing) return
    const id = setTimeout(() => {
      if (s.step >= STORY.length - 1) s.setPlaying(false)
      else s.next()
    }, 7000)
    return () => clearTimeout(id)
  }, [s.playing, s.step, s])

  const A = useMemo(() => readings(s.steps, s.spacing), [s.steps, s.spacing])
  const B = useMemo(
    () => readings(s.compareSteps, s.compareSpacing),
    [s.compareSteps, s.compareSpacing],
  )
  const here = A[Math.min(A.length - 1, Math.max(0, Math.round(s.stepIndex)))]

  const gapIndex = useMemo(() => maxGap(A, B, 'u'), [A, B])
  const gapLambda = useMemo(() => maxGap(A, B, 'lambda'), [A, B])
  const rt = useMemo(() => roundTripError(), [])
  const mono = useMemo(() => monotoneViolations(), [])
  const jumps = useMemo(() => lambdaJumps(), [])
  const perBin = useMemo(() => stepsPerLambdaBin(BINS), [])
  const starved = perBin.filter((c) => c <= 1).length
  const jMed = [...jumps].sort((a, b) => a - b)[Math.floor(jumps.length / 2)]
  const jMax = Math.max(...jumps)
  const onStep = useMemo(() => medianOnStepAxis(), [])
  const lateGrid = onStep.filter((p) => p[0] >= 48).length

  return (
    <div className="page">
      <header>
        <h1>Two clocks for one run</h1>
        <p className="sub">F3 plots against the denoising step, not log-SNR. Here is why.</p>
        <nav>
          <button className={s.mode === 'story' ? 'on' : ''} onClick={() => s.setMode('story')}>
            story
          </button>
          <button className={s.mode === 'explore' ? 'on' : ''} onClick={() => s.setMode('explore')}>
            explore
          </button>
        </nav>
      </header>

      {s.mode === 'story' ? (
        <section className="narration">
          <div className="stepline">
            <button onClick={s.prev} disabled={s.step === 0}>
              ‹
            </button>
            <button onClick={() => s.setPlaying(!s.playing)}>{s.playing ? 'pause' : 'play'}</button>
            <button onClick={s.next} disabled={s.step === STORY.length - 1}>
              ›
            </button>
            <div className="pips">
              {STORY.map((_, i) => (
                <span key={i} className={i === s.step ? 'pip on' : 'pip'} onClick={() => s.goto(i)} />
              ))}
            </div>
            <span className="count">
              {s.step + 1} of {STORY.length}
            </span>
          </div>
          <motion.div key={s.step} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
            <h2>{STORY[s.step].title}</h2>
            <p>{STORY[s.step].narration}</p>
          </motion.div>
        </section>
      ) : (
        <section className="narration">
          <h2>Explore</h2>
          <p>Every panel is live. The checks at the foot recompute as you move things.</p>
        </section>
      )}

      {/* ---------- warm-up ---------- */}
      <motion.section className="panel" animate={{ opacity: on('warmup') }}>
        <h3>Noise is a mixture</h3>
        <div className="row">
          <div>
            <Plot
              width={400}
              height={190}
              xDomain={[0, 1]}
              yDomain={[-3.2, 3.2]}
              xLabel=""
              yLabel=""
              series={[
                {
                  points: noisySignal(1).map((v, i, a) => [i / (a.length - 1), v]),
                  color: LAM,
                  width: 1.6,
                  opacity: 0.45,
                },
                {
                  points: noisySignal(s.alphaBar).map((v, i, a) => [i / (a.length - 1), v]),
                  color: 'var(--ink)',
                  width: 1.2,
                },
              ]}
              inlineLabels={[
                { x: 0.03, y: 2.9, text: 'what you see', color: 'var(--ink)' },
                { x: 0.03, y: -2.9, text: 'the picture underneath', color: '#1b6b70' },
              ]}
            />
          </div>
          <div className="readout">
            <div className="sharebar">
              <div className="sig" style={{ width: `${s.alphaBar * 100}%` }}>
                picture {fmt(s.alphaBar * 100, 0)}%
              </div>
              <div className="noi">fuzz {fmt((1 - s.alphaBar) * 100, 0)}%</div>
            </div>
            <label>
              <span>how much picture is left</span>
              <input
                type="range"
                min={0.0005}
                max={0.9995}
                step={0.0005}
                value={s.alphaBar}
                onChange={(e) => s.patch({ alphaBar: Number(e.target.value) })}
              />
            </label>
            <p className="bignum">
              λ = {fmt(Math.log(s.alphaBar / (1 - s.alphaBar)), 2)}
              <span>log of picture ÷ fuzz</span>
            </p>
            <Details summary="the two lines of maths behind this">
              <Tex block>
                {'x_t=\\sqrt{\\bar\\alpha}\\,x_0+\\sqrt{1-\\bar\\alpha}\\,\\varepsilon'}
              </Tex>
              <Tex block>{'\\lambda=\\log\\frac{\\bar\\alpha}{1-\\bar\\alpha}'}</Tex>
              <p>
                ᾱ is the picture&apos;s share of the power, 1−ᾱ the fuzz&apos;s. So λ is the
                log-odds that a unit of power came from the picture. The log is there because the
                odds span eight orders of magnitude in one run.
              </p>
              <p>The signal drawn here is a fixed sine mix. The noise draw is fixed too, so the
                slider changes only the mixture.</p>
            </Details>
          </div>
        </div>
      </motion.section>

      {/* ---------- schedule ---------- */}
      <motion.section className="panel" animate={{ opacity: on('schedule') }}>
        <h3>1000 noise levels, 50 visited</h3>
        <div className="row">
          <Plot
            xDomain={[0, T_MAX]}
            yDomain={[0, 1]}
            xLabel="noise level t"
            yLabel="picture share"
            xTickFormat={(v) => String(Math.round(v))}
            series={[
              { points: TS.map((t) => [t, ALPHA_BAR[t]]), color: 'var(--ink)', width: 1.8 },
              {
                points: samplerTimesteps(s.steps, s.spacing).map((t) => [t, ALPHA_BAR[t]]),
                color: IDX,
                width: 0,
                dots: 2.4,
              },
            ]}
            markers={[{ x: here.t, y: ALPHA_BAR[here.t], color: IDX, label: `t=${here.t}` }]}
          />
          <div className="readout">
            <p className="bignum">
              {Math.round(s.stepIndex)}
              <span>step, of {s.steps}</span>
            </p>
            <label>
              <span>which step</span>
              <input
                type="range"
                min={0}
                max={s.steps - 1}
                step={1}
                value={Math.round(s.stepIndex)}
                onChange={(e) => s.patch({ stepIndex: Number(e.target.value) })}
              />
            </label>
            <Details summary="the schedule, and where it comes from">
              <p>
                The picture share ᾱ runs 0.999 down to 0.005 over 1000 levels. Orange dots are the
                levels this sampler visits. The cached runs used 50 steps, t = 981 down to 1.
              </p>
              <Plot
                width={330}
                height={150}
                xDomain={[0, T_MAX]}
                yDomain={[0, 0.013]}
                xLabel="noise level t"
                yLabel="β, per-level bite"
                xTickFormat={(v) => String(Math.round(v))}
                yTickFormat={(v) => v.toFixed(3)}
                series={[{ points: TS.map((t) => [t, betaAt(t)]), color: 'var(--ink)', width: 1.6 }]}
              />
              <p className="source">{ALPHA_BAR_SOURCE}. β = 1 − ᾱ(t)/ᾱ(t−1).</p>
            </Details>
          </div>
        </div>
      </motion.section>

      {/* ---------- lambda ---------- */}
      <motion.section className="panel" animate={{ opacity: on('lambda') }}>
        <h3>The same clock, relabelled</h3>
        <div className="row">
          <Plot
            xDomain={[0, T_MAX]}
            yDomain={[-6, 8]}
            xLabel="noise level t"
            yLabel="λ"
            xTickFormat={(v) => String(Math.round(v))}
            series={[{ points: TS.map((t) => [t, lambdaAt(t)]), color: LAM, width: 2 }]}
            markers={[
              { x: here.t, y: lambdaAt(here.t), color: IDX, label: `λ=${fmt(here.lambda)}` },
            ]}
          />
          <div className="readout">
            <p className="lede">λ only ever falls. So every t has its own λ, and you can go back.</p>
            <Details summary="why that matters">
              <p>
                Strictly falling means the relabelling is reversible: no information is added or
                lost by switching axes. Both axes describe the same run. Which to draw on is a
                question about the reader, not about the maths.
              </p>
              <p>Definition taken verbatim from Cell.log_snr() in cache.py.</p>
            </Details>
          </div>
        </div>
      </motion.section>

      {/* ---------- the landing ---------- */}
      <motion.section className="panel" animate={{ opacity: on('landing') }}>
        <h3>Even in steps, uneven in λ</h3>
        <div className="row">
          <Plot
            xDomain={[0, 49]}
            yDomain={[0, 2.6]}
            xLabel="step"
            xColor="#b5561f"
            yLabel="λ moved"
            xTickFormat={(v) => String(Math.round(v))}
            series={[
              {
                points: jumps.map((j, i) => [i + 1, j] as [number, number]),
                color: IDX,
                width: 1.6,
                dots: 2,
              },
            ]}
            hlines={[{ y: jMed, color: INK }]}
            inlineLabels={[
              { x: 12, y: 0.42, text: `most steps: ${fmt(jMed, 2)}`, color: '#6d665c' },
              { x: 30, y: 2.3, text: `last step: ${fmt(jMax, 2)}`, color: '#b5561f' },
            ]}
          />
          <div className="readout">
            <p className="bignum">
              {fmt(jMax / jMed, 0)}×
              <span>further, on the last step alone</span>
            </p>
            <p className="lede">
              Put 20 evenly spaced λ points over that run. {starved} of them have one real step
              under them, or none.
            </p>
            <div className="bins">
              {perBin.map((c, i) => (
                <div key={i} className={c <= 1 ? 'bin thin' : 'bin'} title={`${c} steps`}>
                  <div className="bar" style={{ height: `${6 + c * 9}px` }} />
                </div>
              ))}
            </div>
            <p className="legend">noisy left, clean right. Orange bars are the starved ones.</p>
            <Details summary="why this is the cost of the λ axis">
              <p>
                The run really does spend its last step crossing a fifth of the λ range. Any curve
                drawn there is interpolation between two measurements, drawn as wide as the region
                it spans. That invites the reader to read detail nobody measured.
              </p>
              <p className="source">Timesteps read from {CACHED_TIMESTEPS_SOURCE}.</p>
            </Details>
          </div>
        </div>
      </motion.section>

      {/* ---------- the same values on each axis ---------- */}
      <motion.section className="panel" animate={{ opacity: on('axes') }}>
        <h3>One curve, drawn on each clock</h3>
        <div className="row">
          <div>
            <Plot
              xDomain={[F3_LO, F3_HI]}
              yDomain={[0.5, 1.6]}
              xLabel="λ"
              xColor="#1b6b70"
              yLabel="correction size"
              bands={[
                {
                  points: F3.log_snr_grid.map((g: number, i: number) => [
                    g,
                    F3.median_curve[i] - F3.iqr[i] / 2,
                    F3.median_curve[i] + F3.iqr[i] / 2,
                  ]),
                  color: '#1b6b70',
                  opacity: 0.16,
                },
              ]}
              series={[
                {
                  points: F3.log_snr_grid.map((g: number, i: number) => [g, F3.median_curve[i]]),
                  color: LAM,
                  width: 2.4,
                  dots: 2.2,
                },
              ]}
              hlines={[{ y: 1, color: INK }]}
            />
          </div>
          <div>
            <Plot
              xDomain={[0, 49]}
              yDomain={[0.5, 1.6]}
              xLabel="step"
              xColor="#b5561f"
              yLabel="same values"
              series={[{ points: onStep, color: IDX, width: 2.4, dots: 2.2 }]}
              hlines={[{ y: 1, color: INK }]}
              xTickFormat={(v) => String(Math.round(v))}
            />
          </div>
        </div>
        <p className="lede">
          Same twenty numbers. The wide rise on the left is {lateGrid} points sitting on the last
          two steps.
        </p>
        <Details summary="the numbers behind this curve, and what it may not claim">
          <table className="nums">
            <tbody>
              <tr>
                <td>pairs / curves</td>
                <td>
                  {F3.n_pairs} / {F3.n_curves}
                </td>
              </tr>
              <tr>
                <td>λ range covered</td>
                <td>
                  {fmt(F3_LO)} to {fmt(F3_HI)}
                </td>
              </tr>
              <tr className="hi">
                <td>spread of curves about the median</td>
                <td>{fmt(F3.collapse_spread_pct, 1)}%</td>
              </tr>
              <tr>
                <td>the script&apos;s verdict on that spread</td>
                <td>{F3.verdict}</td>
              </tr>
              <tr>
                <td>peak of the median</td>
                <td>
                  λ={fmt(F3.peak_log_snr)} {F3.peak_at_edge ? '(at the edge, still rising)' : ''}
                </td>
              </tr>
            </tbody>
          </table>
          <p>
            The band is the middle half of the curves, not a confidence interval. Neither panel
            shows that the curves lie on top of each other: {fmt(F3.collapse_spread_pct, 1)}% is
            loose by the script&apos;s own thresholds, and the peak at the right edge means the
            grid stopped before the curve did.
          </p>
          <p>
            The step panel is the published median with x relabelled, not a recomputation from
            per-cell values. snr_collapse.py saves only the median and the band.
          </p>
          <p className="source">
            {F3_PATH}. Normalisation: {F3.normalize}.
          </p>
        </Details>
      </motion.section>

      {/* ---------- the cost ---------- */}
      <motion.section className="panel" animate={{ opacity: on('compare') }}>
        <h3>What the step axis costs</h3>
        <div className="row">
          <Plot
            xDomain={[0, 1]}
            yDomain={[0.55, 1.5]}
            xLabel="step, rescaled"
            xColor="#b5561f"
            yLabel="correction size"
            series={[
              { points: A.map((r) => [r.u, r.y]), color: IDX, width: 2, dots: 2 },
              { points: B.map((r) => [r.u, r.y]), color: IDX, width: 1.6, dash: '5 4', dots: 2 },
            ]}
            inlineLabels={[{ x: 0.42, y: 1.42, text: 'two samplers, apart', color: '#b5561f' }]}
          />
          <Plot
            xDomain={[F3_LO, F3_HI]}
            yDomain={[0.55, 1.5]}
            xLabel="λ"
            xColor="#1b6b70"
            yLabel=""
            series={[
              { points: A.map((r) => [r.lambda, r.y]), color: LAM, width: 2, dots: 2 },
              {
                points: B.map((r) => [r.lambda, r.y]),
                color: LAM,
                width: 1.6,
                dash: '5 4',
                dots: 2,
              },
            ]}
            inlineLabels={[{ x: -3.5, y: 1.42, text: 'same two, together', color: '#1b6b70' }]}
          />
        </div>
        <div className="gapbox">
          <div>
            <span className="val" style={{ color: 'var(--idx)' }}>
              {fmt(gapIndex, 2)}
            </span>
            <span className="lab">apart, on the step axis</span>
          </div>
          <div>
            <span className="val" style={{ color: 'var(--lam)' }}>
              {fmt(gapLambda, 2)}
            </span>
            <span className="lab">apart, on the λ axis</span>
          </div>
        </div>
        <p className="lede">
          That is the λ axis&apos;s one advantage. It pays off only with a second sampler, and this
          paper runs one.
        </p>
        <Details summary="the same point without diffusion in it">
          <p>
            Two people walk one trail. One notes the altitude at each of their 50 rest stops, the
            other at each of their 20. Plot altitude against rest-stop number and the graphs
            disagree, because a rest stop is not a fixed amount of trail. Plot against distance
            from the trailhead and they coincide.
          </p>
          <p>
            Step index is the rest stop, λ is the distance. The catch F3 hit: if one walker sprints
            the last kilometre, the distance axis draws that sprint wide and empty.
          </p>
          <p>
            Each sampler here is F3&apos;s measured median re-sampled at its own timesteps, not a
            separate generation run. Leading and trailing are the two real DDIM spacings;
            evenly-spaced-in-λ is idealised, standing in for the EDM/Karras family.
          </p>
        </Details>
        <Details summary="change the two samplers">
          <div className="ctlgrid">
            <label>
              <span>sampler A steps</span>
              <select
                value={s.steps}
                onChange={(e) => s.patch({ steps: Number(e.target.value), stepIndex: 0 })}
              >
                {[20, 30, 50, 100].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>sampler A spacing</span>
              <select
                value={s.spacing}
                onChange={(e) => s.patch({ spacing: e.target.value as Spacing, stepIndex: 0 })}
              >
                {Object.entries(SPACING_LABEL).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>sampler B steps</span>
              <select
                value={s.compareSteps}
                onChange={(e) => s.patch({ compareSteps: Number(e.target.value) })}
              >
                {[20, 30, 50, 100].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>sampler B spacing</span>
              <select
                value={s.compareSpacing}
                onChange={(e) => s.patch({ compareSpacing: e.target.value as Spacing })}
              >
                {Object.entries(SPACING_LABEL).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p>
            Set both to the same steps and spacing and both numbers go to zero. The λ number is not
            zero otherwise, because joining a sampler&apos;s samples across a stretch of λ it never
            stepped in is itself an error.
          </p>
        </Details>
      </motion.section>

      {/* ---------- checks ---------- */}
      <section className="panel checks">
        <h3>Checks, run in the page</h3>
        <ul>
          <li>
            <span className={mono === 0 ? 'ok' : 'bad'}>{mono === 0 ? 'holds' : 'fails'}</span>
            λ falls at every one of the {T_MAX} levels. Violations: {mono}.
          </li>
          <li>
            <span className={rt < 1e-6 ? 'ok' : 'bad'}>{rt < 1e-6 ? 'holds' : 'fails'}</span>
            λ converts back to the picture share exactly. Worst error: {rt.toExponential(1)}.
          </li>
          <li>
            <span className="ok">holds</span>
            {starved} of {BINS} λ points sit on one real step or none.
          </li>
          <li>
            <span className="note">read from disk</span>
            F3&apos;s {fmt(F3.collapse_spread_pct, 1)}% spread. Measured elsewhere, not checked
            here.
          </li>
        </ul>
        <Details summary="what each check is testing">
          <p>
            Falling λ means step and λ carry the same information, so the axis swap is reversible.
            The round trip is σ(λ) = ᾱ against the scheduler&apos;s own table: at the current step,
            σ({fmt(here.lambda, 3)}) = {fmt(alphaBarFromLambda(here.lambda), 6)} against{' '}
            {fmt(ALPHA_BAR[here.t], 6)}. The starved count is the argument for the step axis, taken
            from the {CACHED.length} timesteps the cached runs really visited.
          </p>
        </Details>
      </section>

      <footer>
        <p>
          F3 of <code>paper/iclr/what-each-figure-argues.md</code>. Curve from{' '}
          <code>scripts/snr_collapse.py</code>. Full notes in the README.
        </p>
      </footer>
    </div>
  )
}
