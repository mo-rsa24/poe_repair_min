// One view per claim. Every panel reads the shared state, so moving a scrubber
// moves the number and the picture together.
import { scaleLinear } from 'd3-scale'
import {
  Band, Cannot, Frame, H, Legend, Line, Marker, M, plainText, Plot, Quoted, Src, Stat, W,
} from './parts'
import { useShallow } from 'zustand/react/shallow'
import { D, expertCell, nearestRow, rowNearestLogSnr, useScene } from './store'
import type { ClimbCell } from './types'

const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const f2 = (x: number) => x.toFixed(2)
const f3 = (x: number) => x.toFixed(3)

// ------------------------------------------------------------------ shared

/** The five views at one point in the run, for the pairs that have them.
 *  All five come from the one latent the PoE run was carrying at that step, so
 *  they are five readings of one state, not five runs. */
function ExpertStrip({ step, caption }: { step: number; caption: string }) {
  const { pair, seed, set } = useScene(useShallow((s) => ({ pair: s.pair, seed: s.seed, set: s.set })))
  const cell = expertCell(pair, seed)
  if (!cell) {
    return <div className="slot">no decoded frames for {pair} seed {seed}</div>
  }
  const row = nearestRow(cell, step)
  const label: Record<string, string> = {
    uncond: 'no prompt', a: cell.promptA, b: cell.promptB,
    poe: 'PoE', joint: 'joint prompt',
  }
  // All five views are formed from the SAME latent, the one the PoE run was
  // carrying at this step, so they can only differ by the size of the noise
  // still in it. Late in the run that is almost nothing and the five pictures
  // converge on the PoE image no matter what the prompt asks for.
  const late = row.step > cell.nSteps * 0.6
  return (
    <div>
      <div className="ctl">
        <label>pair and seed</label>
        <select value={`${cell.pair}|${cell.seed}`}
                onChange={(e) => {
                  const [p, s] = e.target.value.split('|')
                  set({ pair: p, seed: Number(s) })
                }}>
          {D.experts.cells.map((c) => (
            <option key={`${c.pair}|${c.seed}`} value={`${c.pair}|${c.seed}`}>
              {c.pair.replace(/_/g, ' ')} · seed {c.seed}
            </option>
          ))}
        </select>
        <span className="val">step {row.step} of {cell.nSteps}</span>
      </div>
      <div className="frames">
        {D.experts.views.map((v) => (
          <Frame key={v} src={row.views[v]} cap={label[v] ?? v}
                 kind={v === 'poe' ? 'poe' : v === 'joint' ? 'mono' : undefined} />
        ))}
      </div>
      <div className="note">
        <b>These are not five separate runs.</b> Every panel is the same latent,
        the one PoE was carrying at step {row.step}, asked what finished picture
        it implies under a different prompt. So the cat panel is not a cat: it is
        what the cat prompt makes of the state PoE has already reached, and the
        joint panel is the correction's target direction, not the image the joint
        prompt would have produced on its own.
      </div>
      {late && (
        <div className="note bad">
          <b>Past the middle of the run these five must coincide.</b> The estimate
          is the latent minus a shrinking multiple of the prediction, and by step
          {' '}{row.step} of {cell.nSteps} that multiple is nearly zero, so every
          prompt returns the picture PoE has already committed to. Drag back
          towards step 0 to see them actually differ.
        </div>
      )}
      <div className="src">{caption} · frames every 5 steps, nearest shown</div>
      <Src keys={['experts_index']} />
    </div>
  )
}

/** Which of the three arms a claim actually has data for. Drawing an arm that
 *  was never sampled would be inventing it, so an absent arm says so. */
function Arms({ poe, mono, lora }: { poe: string; mono: string; lora: string }) {
  return (
    <table>
      <thead><tr><th>arm</th><th>what this claim has</th></tr></thead>
      <tbody>
        <tr><td style={{ color: 'var(--poe)' }}>PoE</td><td>{poe}</td></tr>
        <tr><td style={{ color: 'var(--mono)' }}>Mono</td><td>{mono}</td></tr>
        <tr><td style={{ color: 'var(--lora)' }}>LoRA</td><td>{lora}</td></tr>
      </tbody>
    </table>
  )
}

// ------------------------------------------------- C1: size vs noise level

export function C1() {
  const { bin, set } = useScene(useShallow((s) => ({ bin: s.bin, set: s.set })))
  const c = D.collapse.prereg
  const x = scaleLinear().domain([c.grid[0], c.grid[c.grid.length - 1]]).range([M.l, W - M.r])
  const hi = c.median.map((m, i) => m + c.iqr[i] / 2)
  const lo = c.median.map((m, i) => Math.max(0, m - c.iqr[i] / 2))
  const y = scaleLinear().domain([0, Math.max(...hi) * 1.05]).range([H - M.b, M.t]).nice()
  const at = c.grid[bin]

  return (
    <>
      <div className="row">
        <div className="grow panel">
          <h3>The committed size measure, median over {c.nCurves} curves</h3>
          <Plot x={x} y={y} xLabel="log-SNR (early run on the left)"
                yLabel="correction size, own-median scaled">
            <Band xs={c.grid} lo={lo} hi={hi} x={x} y={y} color="var(--poe)" />
            <Line xs={c.grid} ys={c.median} x={x} y={y} color="var(--poe)" />
            <Marker x={x} at={at} />
          </Plot>
          <div className="ctl">
            <label>log-SNR</label>
            <input type="range" min={0} max={c.grid.length - 1} value={bin}
                   onChange={(e) => set({ bin: Number(e.target.value) })} />
            <span className="val">{f2(at)}</span>
          </div>
          <Stat items={[
            { label: 'median here', value: f2(c.median[bin]) },
            { label: 'spread across pairs', value: `${c.spreadPct.toFixed(1)}%` },
            { label: 'reading', value: c.verdict, tone: 'hold' },
            { label: 'pairs / curves', value: `${c.nPairs} / ${c.nCurves}` },
          ]} />
          <div className="note">
            <b>The collapse is loose.</b> One shared curve would mean the correction
            is a property of the noise level, not of the particular animals. At
            {' '}{c.spreadPct.toFixed(1)}% spread it is only partly that. The caption
            may claim the collapse no further than this number supports.
          </div>
          <Src keys={['collapse_prereg']} />
        </div>

        <div className="grow panel">
          <h3>What the run looks like at that noise level</h3>
          <ExpertStripAtLogSnr logSnr={at} />
        </div>
      </div>

      <div className="panel">
        <h3>Which arms this claim has</h3>
        <Arms
          poe="the cached path, 34 curves over 17 pairs"
          mono="not measured here: the size is of the correction to PoE, so there is no second path in this analysis"
          lora="not measured here"
        />
      </div>

      <Cannot>
        whether the size peaks anywhere, because the committed measure is still
        rising at the right edge. It also cannot show individual pairs: only the
        median and the spread band were written to disk, not the 34 curves.
      </Cannot>
    </>
  )
}

/** C1's scrubber is in log-SNR, so the pictures follow log-SNR, not step. */
function ExpertStripAtLogSnr({ logSnr }: { logSnr: number }) {
  const { pair, seed } = useScene(useShallow((s) => ({ pair: s.pair, seed: s.seed })))
  const cell = expertCell(pair, seed)
  if (!cell) return <div className="slot">no decoded frames for this pair</div>
  const row = rowNearestLogSnr(cell, logSnr)
  return <ExpertStrip step={row.step} caption={`nearest decoded log-SNR ${f2(row.logSnr)}`} />
}

// ------------------------------------------- C2: the two measures disagree

export function C2() {
  const { bin, measure, set } = useScene(useShallow((s) => ({ bin: s.bin, measure: s.measure, set: s.set })))
  const a = D.collapse.prereg
  const b = D.collapse.raw
  const x = scaleLinear().domain([a.grid[0], a.grid[a.grid.length - 1]]).range([M.l, W - M.r])
  const all = [...a.median, ...b.median]
  const y = scaleLinear().domain([0, Math.max(...all) * 1.05]).range([H - M.b, M.t]).nice()
  const at = a.grid[bin]

  return (
    <>
      <div className="note">
        <b>One knob differs</b> between these two curves: how the correction's size
        is normalized. Same 17 pairs, same 34 cells, same run. So the gap between
        them is the measure and nothing else.
      </div>

      <div className="row">
        <div className="grow panel">
          <h3>Committed measure against raw size</h3>
          <div className="ctl">
            <label>show</label>
            <div className="seg">
              {(['both', 'prereg', 'raw'] as const).map((m) => (
                <button key={m} className={measure === m ? 'on' : ''}
                        onClick={() => set({ measure: m })}>
                  {m === 'prereg' ? 'committed' : m === 'raw' ? 'raw' : 'both'}
                </button>
              ))}
            </div>
          </div>
          <Plot x={x} y={y} xLabel="log-SNR" yLabel="size, own-median scaled">
            {measure !== 'raw' && (
              <Line xs={a.grid} ys={a.median} x={x} y={y} color="var(--poe)" />
            )}
            {measure !== 'prereg' && (
              <Line xs={b.grid} ys={b.median} x={x} y={y} color="var(--lora)" dash="5 4" />
            )}
            {measure !== 'prereg' && !b.peakAtEdge && (
              <line x1={x(b.peakLogSnr)} x2={x(b.peakLogSnr)} y1={M.t} y2={H - M.b}
                    stroke="var(--lora)" strokeWidth={1.5} />
            )}
            <Marker x={x} at={at} />
          </Plot>
          <Legend items={[
            { color: 'var(--poe)', label: 'committed: size relative to the prediction' },
            { color: 'var(--lora)', label: 'raw: size of the correction alone' },
          ]} />
          <table style={{ marginTop: 14 }}>
            <thead><tr><th>measure</th><th className="num">spread</th><th>peak</th></tr></thead>
            <tbody>
              <tr>
                <td>committed</td>
                <td className="num">{a.spreadPct.toFixed(1)}%</td>
                <td>{a.peakAtEdge ? 'still rising at the right edge, no interior peak' : f2(a.peakLogSnr)}</td>
              </tr>
              <tr>
                <td>raw</td>
                <td className="num">{b.spreadPct.toFixed(1)}%</td>
                <td>{b.peakAtEdge ? 'at the edge' : `log-SNR ${f2(b.peakLogSnr)}`}</td>
              </tr>
            </tbody>
          </table>
          <div className="note">
            <b>They disagree about where the peak is.</b> The raw size peaks at
            log-SNR {f2(b.peakLogSnr)}; the committed measure never turns over.
            The difference is the denominator: the size of the PoE prediction
            itself falls along the run, which lifts the committed ratio at the
            end. Neither curve is the timing answer. Timing is plan 04's question.
          </div>
          <Src keys={['collapse_prereg', 'collapse_raw']} />
        </div>

        <div className="grow panel">
          <h3>The run at the noise level you are pointing at</h3>
          <ExpertStripAtLogSnr logSnr={at} />
        </div>
      </div>

      <Cannot>
        which measure is the right one. It shows that the choice changes the
        answer, which is why the committed one was fixed before the run.
      </Cannot>
    </>
  )
}

// ------------------------------------------------------ C3: where they fork

export function C3() {
  const { step, forkPair, forkSeed, forkCoverage, set } = useScene(useShallow((s) => ({
    step: s.step, forkPair: s.forkPair, forkSeed: s.forkSeed,
    forkCoverage: s.forkCoverage, set: s.set,
  })))
  const read = D.fork[forkCoverage]
  const cell =
    read.cells.find((c) => c.pair === forkPair && c.seed === forkSeed) ?? read.cells[0]
  const img = D.forkImages.find((i) => i.pair === cell.pair && i.seed === cell.seed)

  const x = scaleLinear().domain([0, cell.distance.length - 1]).range([M.l, W - M.r])
  const y = scaleLinear().domain([0, Math.max(...cell.distance) * 1.05]).range([H - M.b, M.t]).nice()
  const s = Math.min(step, cell.distance.length - 1)

  return (
    <>
      <div className="row">
        <div className="grow panel">
          <h3>Distance between the two paths, step by step</h3>
          <div className="ctl">
            <label>cell</label>
            <select value={`${cell.pair}|${cell.seed}`}
                    onChange={(e) => {
                      const [p, sd] = e.target.value.split('|')
                      set({ forkPair: p, forkSeed: Number(sd) })
                    }}>
              {read.cells.map((c) => (
                <option key={`${c.pair}|${c.seed}`} value={`${c.pair}|${c.seed}`}>
                  {c.pair.replace(/_/g, ' ')} · seed {c.seed} · elbow {c.elbowStep}
                </option>
              ))}
            </select>
          </div>
          <Plot x={x} y={y} xLabel="sampling step" yLabel="distance between paths">
            <line x1={x(cell.elbowStep)} x2={x(cell.elbowStep)} y1={M.t} y2={H - M.b}
                  stroke="var(--mono)" strokeWidth={1.5} />
            <Line xs={cell.distance.map((_, i) => i)} ys={cell.distance} x={x} y={y}
                  color="var(--poe)" />
            <Marker x={x} at={s} />
          </Plot>
          <div className="ctl">
            <label>step</label>
            <input type="range" min={0} max={cell.distance.length - 1} value={s}
                   onChange={(e) => set({ step: Number(e.target.value) })} />
            <span className="val">{s}</span>
          </div>
          <Stat items={[
            { label: 'distance here', value: f2(cell.distance[s]) },
            { label: 'this cell elbow', value: String(cell.elbowStep) },
            { label: 'median elbow', value: String(read.medianElbow) },
            { label: 'cells', value: String(read.nCells) },
          ]} />
          <Src keys={[forkCoverage === 'refreshed' ? 'fork_43cells' : 'fork_19cells']} />
        </div>

        <div className="grow panel">
          <h3>The two paths, where they end up</h3>
          <div className="frames">
            <Frame src={img?.poe ?? null} cap="PoE, no correction" kind="poe" wide />
            <Frame src={img?.mono ?? null} cap="Mono, corrected" kind="mono" wide />
          </div>
          <div className="note">
            <b>These are the final frames, not step {s}.</b> The saved trajectories
            hold only the noisy latent, so decoding them at the elbow shows noise.
            The picture that would move with the scrubber is gap G1 on the Owed
            page, with the run that produces it.
          </div>
          <div className="src">
            {img?.poe ?? 'no frame'}
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Coverage: the review reads 19 cells, the disk has 43</h3>
        <div className="ctl">
          <label>read over</label>
          <div className="seg">
            <button className={forkCoverage === 'refreshed' ? 'on' : ''}
                    onClick={() => set({ forkCoverage: 'refreshed' })}>
              43 cells (every eligible cell)
            </button>
            <button className={forkCoverage === 'original' ? 'on' : ''}
                    onClick={() => set({ forkCoverage: 'original' })}>
              19 cells (the review's run)
            </button>
          </div>
        </div>
        <table>
          <thead>
            <tr><th>read</th><th className="num">cells</th><th className="num">median elbow</th>
                <th className="num">in steps 13 to 20</th><th className="num">range</th>
                <th className="num">distance at step 0</th></tr>
          </thead>
          <tbody>
            {(['original', 'refreshed'] as const).map((kk) => {
              const r = D.fork[kk]
              return (
                <tr key={kk} className={forkCoverage === kk ? 'on' : ''}>
                  <td>{kk === 'original' ? "the review's 19" : 'every eligible cell'}</td>
                  <td className="num">{r.nCells}</td>
                  <td className="num">{r.medianElbow}</td>
                  <td className="num">{r.inBand13to20} of {r.nCells}</td>
                  <td className="num">{r.elbowMin} to {r.elbowMax}</td>
                  <td className="num">{r.maxDistanceAtZero.toFixed(2)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="note">
          <b>The elbow holds at step {D.fork.refreshed.medianElbow} over
          {' '}{D.fork.refreshed.nCells} cells</b>, up from {D.fork.original.nCells}.
          Distance at step 0 is {D.fork.refreshed.maxDistanceAtZero.toFixed(2)} on
          every cell, which is the check that both paths start from the same
          pinned noise: if that were not zero the fork would be measuring the
          initialisation, not the correction.
        </div>
        <Src keys={['fork_19cells', 'fork_43cells']} />
      </div>

      <div className="panel">
        <h3>Which arms this claim has</h3>
        <Arms
          poe="lam000, a full 51-step path per cell"
          mono="lam100, a full 51-step path per cell"
          lora="no LoRA path was sampled for these cells, so there is no third curve to draw"
        />
      </div>

      <Cannot>
        why they separate there, only that they do. It also cannot show the
        pictures mid-run: see gap G1.
      </Cannot>
    </>
  )
}

// ------------------------------------------------------------ C4: not run

export function C4() {
  const gap = D.gaps.find((g) => g.closes === 'C4')!
  return (
    <div className="panel">
      <h3>Empty slot: this comparison has no run behind it</h3>
      <div className="slot" style={{ padding: 40 }}>
        the fork step, {D.fork.refreshed.medianElbow}, against the window plan 04
        measures
        <br /><br />
        nothing is drawn here because nothing has been measured
      </div>
      <div className="note">
        <b>What would fill it:</b> {gap.what}
        <br /><b>Why it is empty:</b> {gap.why}
      </div>
      <pre>{gap.command}</pre>
      <div className="src">output would land in {gap.output} · cost: {gap.cost}</div>
    </div>
  )
}

// -------------------------------------------------------- C5: the climb

export function C5() {
  const { step, set } = useScene(useShallow((s) => ({ step: s.step, set: s.set })))
  const c = D.climb
  const cells = c.cells

  const strips: { key: keyof ClimbCell; label: string; color: string }[] = [
    { key: 'normalised', label: 'the correction', color: 'var(--poe)' },
    { key: 'controlWrongStep', label: 'control: right correction, wrong step', color: 'var(--control-b)' },
    { key: 'controlRandom', label: 'control: a random vector', color: 'var(--control-a)' },
    { key: 'rVsEpsPoe', label: 'the correction against the PoE prediction', color: 'var(--mono)' },
  ]
  const values = strips.map((s2) => cells.map((cc) => cc[s2.key] as number))
  const lo = Math.min(-0.3, ...values.flat())
  const hi = Math.max(...values.flat(), 0.6)
  // Right margin left clear for the median readout, so it never sits on a dot.
  const x = scaleLinear().domain([lo, hi]).range([M.l, W - M.r - 52])

  const withCosine = cells.filter((cc) => cc.perStepCosine.length > 0)
  const cos = withCosine[0]
  const cx = cos ? scaleLinear().domain([0, cos.perStepCosine.length - 1]).range([M.l, W - M.r]) : null
  const cy = scaleLinear().domain([0, 1]).range([H - M.b, M.t])

  return (
    <>
      <div className="row">
        <div className="grow panel">
          <h3>Where each cell lands, with both controls as siblings</h3>
          <svg width={W} height={30 + strips.length * 46} role="img">
            <line x1={x(0)} x2={x(0)} y1={8} y2={strips.length * 46 + 4}
                  stroke="var(--line)" strokeWidth={1} />
            {strips.map((s2, si) => {
              const yy = 24 + si * 46
              const vals = values[si]
              const med = [...vals].sort((p, q) => p - q)[Math.floor(vals.length / 2)]
              return (
                <g key={s2.key as string}>
                  <text x={M.l} y={yy - 9} fontSize="11" fill="var(--ink-dim)">{s2.label}</text>
                  {vals.map((v, i) => (
                    <circle key={i} cx={x(v)} cy={yy + 9} r={3} fill={s2.color} opacity={0.5} />
                  ))}
                  <line x1={x(med)} x2={x(med)} y1={yy + 1} y2={yy + 17}
                        stroke={s2.color} strokeWidth={2} />
                  <text x={W - M.r} y={yy + 13} textAnchor="end" fontSize="11"
                        fill={s2.color} fontFamily="var(--mono-font)">
                    {med >= 0 ? '+' : ''}{f3(med)}
                  </text>
                </g>
              )
            })}
            <text x={x(0)} y={strips.length * 46 + 20} textAnchor="middle" fontSize="10"
                  fill="var(--ink-faint)">0</text>
          </svg>
          <Stat items={[
            { label: 'cells', value: String(c.nCells) },
            { label: 'cells pushing backwards', value: String(c.nNegative), tone: c.nNegative === 0 ? 'pass' : 'fail' },
            { label: 'median push', value: `+${f3(c.medians.normalised)}` },
            { label: 'random control', value: `+${f3(c.medians.controlRandom)}` },
          ]} />
          <div className="note">
            <b>Read the sign carefully.</b> A sampling step moves along minus the
            prediction, and the correction sits on the opposite side of zero from
            the prediction (median {f3(c.medians.rVsEpsPoe)}). So the correction
            subtracts from what PoE asks for. That is what "PoE overshoots into a
            blend" predicts.
          </div>
          <Src keys={['climb_38cells']} />
        </div>

        <div className="grow panel">
          <h3>How the alignment decays through the run</h3>
          {cos && cx ? (
            <>
              <Plot x={cx} y={cy} xLabel="sampling step" yLabel="cosine with the motion">
                {withCosine.map((cc, i) => (
                  <Line key={i} xs={cc.perStepCosine.map((_, j) => j)} ys={cc.perStepCosine}
                        x={cx} y={cy} color="var(--poe)" width={0.7} />
                ))}
                <Marker x={cx} at={Math.min(step, cos.perStepCosine.length - 1)} />
              </Plot>
              <div className="ctl">
                <label>step</label>
                <input type="range" min={0} max={cos.perStepCosine.length - 1} value={Math.min(step, cos.perStepCosine.length - 1)}
                       onChange={(e) => set({ step: Number(e.target.value) })} />
                <span className="val">{Math.min(step, cos.perStepCosine.length - 1)}</span>
              </div>
              <Stat items={[{
                label: `median at step ${Math.min(step, cos.perStepCosine.length - 1)}`,
                value: f2(medianAt(withCosine, Math.min(step, cos.perStepCosine.length - 1))),
              }]} />
            </>
          ) : (
            <div className="slot">no per-step cosine was saved</div>
          )}
          <h3>The same step, in pictures</h3>
          <ExpertStrip step={step} caption="along the PoE path" />
        </div>
      </div>

      <div className="panel">
        <h3>Which arms this claim has</h3>
        <Arms
          poe="the cached path, all 38 cells"
          mono={c.caveat}
          lora="not measured here"
        />
        <div className="note">
          <b>Two files answer this question over different populations.</b> The
          one the review quotes has {c.nCells} cells over {c.nPairs} pairs; the
          other has {c.otherPopulation.nCells} over {c.otherPopulation.nPairs}.
          Both are shown, never averaged.
        </div>
        <Src keys={['climb_38cells', 'climb_34cells']} />
      </div>

      <Cannot>
        what happens along a corrected path. The cache walks the PoE path only,
        so this is the push measured at the states PoE visits.
      </Cannot>
    </>
  )
}

function medianAt(cells: ClimbCell[], step: number) {
  const v = cells.map((c) => c.perStepCosine[step]).filter((n) => n !== undefined)
  const s = [...v].sort((a, b) => a - b)
  return s[Math.floor(s.length / 2)] ?? 0
}

// ------------------------------------------------------- C6: the spectrum

export function C6() {
  const { k, set } = useScene(useShallow((s) => ({ k: s.k, set: s.set })))
  const sp = D.spectrum
  const i = sp.ks.indexOf(k) >= 0 ? sp.ks.indexOf(k) : 3
  const x = scaleLinear().domain([0, sp.ks.length - 1]).range([M.l, W - M.r])
  const y = scaleLinear().domain([0, 0.7]).range([H - M.b, M.t]).nice()
  // The bar is the ratio to the matched floor, recomputed here from the loaded
  // numbers rather than copied from the review file.
  const ratio = sp.energy[i] / sp.floor[i]

  return (
    <>
      <div className="row">
        <div className="grow panel">
          <h3>Energy carried by the first k directions, against a matched floor</h3>
          <Plot x={x} y={y} xLabel="k (rank), doubling each tick" yLabel="share of total energy">
            <Band xs={sp.ks.map((_, j) => j)} lo={sp.floor.map(() => 0)} hi={sp.floor}
                  x={x} y={y} color="var(--floor)" />
            <Line xs={sp.ks.map((_, j) => j)} ys={sp.floor} x={x} y={y} color="var(--floor)" dash="4 3" />
            <Line xs={sp.ks.map((_, j) => j)} ys={sp.energy} x={x} y={y} color="var(--poe)" />
            <Line xs={sp.ks.map((_, j) => j)} ys={sp.heldout} x={x} y={y} color="var(--control-b)" />
            <Marker x={x} at={i} />
          </Plot>
          <div className="ctl">
            <label>rank k</label>
            <input type="range" min={0} max={sp.ks.length - 1} value={i}
                   onChange={(e) => set({ k: sp.ks[Number(e.target.value)] })} />
            <span className="val">k = {sp.ks[i]}</span>
          </div>
          <Legend items={[
            { color: 'var(--poe)', label: 'training pairs' },
            { color: 'var(--control-b)', label: 'held-out pairs, projected into the fitted subspace' },
            { color: 'var(--floor)', label: 'same-shape random floor' },
          ]} />
          <Stat items={[
            { label: `energy at k=${sp.ks[i]}`, value: pct(sp.energy[i]) },
            { label: 'random floor', value: pct(sp.floor[i]) },
            { label: 'ratio to floor', value: `${ratio.toFixed(1)}x`, tone: ratio > 3 ? 'pass' : 'hold' },
            { label: 'held-out', value: pct(sp.heldout[i]) },
          ]} />
          <div className="note">
            <b>Read the ratio, never the raw percentage.</b> With
            {' '}{sp.trainVectors} vectors in {sp.dims.toLocaleString()} dimensions,
            the energy at k is partly forced by how many vectors were stacked. The
            floor is what that forcing looks like on random data of the same shape,
            and it is computed in this page from the loaded numbers.
          </div>
          <Src keys={['spectrum']} />
        </div>

        <div className="grow panel">
          <h3>The singular values themselves</h3>
          <SingularPlot />
          <div className="note">
            <b>Open question, not decided here:</b> {plainText(D.openQuestion.body.join(' ')).replace(/^⚠️ /, '')}
          </div>
          <Quoted q={D.openQuestion} />
        </div>
      </div>

      <Cannot>
        whether a rank-k adapter would work. It measures the geometry of the
        stacked corrections, and C7 shows that geometry and behaviour disagree.
      </Cannot>
    </>
  )
}

function SingularPlot() {
  const sv = D.spectrum.singularValues
  const x = scaleLinear().domain([0, sv.length - 1]).range([M.l, W - M.r])
  const y = scaleLinear().domain([0, Math.max(...sv) * 1.05]).range([H - M.b, M.t]).nice()
  return (
    <Plot x={x} y={y} xLabel="index" yLabel="singular value">
      <Line xs={sv.map((_, i) => i)} ys={sv} x={x} y={y} color="var(--lora)" />
    </Plot>
  )
}

// --------------------------------------------- C7: geometry against transfer

export function C7() {
  const t = D.spectrum.transfer
  const pairs = t.perPair
  // Axis from the data, so no point can fall off the plot unnoticed.
  const gmax = Math.max(...pairs.map((p) => p.geometryK64))
  const x = scaleLinear().domain([0, gmax * 1.15]).range([M.l, W - M.r]).nice()
  const y = scaleLinear().domain([0, 1.05]).range([H - M.b, M.t])

  return (
    <>
      <div className="row">
        <div className="grow panel">
          <h3>What the geometry predicts against what the adapter did</h3>
          <Plot x={x} y={y} xLabel="share of the correction inside the fitted subspace (k=64)"
                yLabel="adapter compose rate">
            {pairs.map((p) => (
              <g key={p.pair}>
                <circle cx={x(p.geometryK64)} cy={y(p.composeRate)} r={5}
                        fill="var(--lora)" opacity={0.85} />
              </g>
            ))}
            <line x1={M.l} x2={W - M.r} y1={y(0)} y2={y(0)} stroke="var(--poe)" strokeWidth={2} />
            <text x={W - M.r - 4} y={y(0) - 6} textAnchor="end" fontSize="10" fill="var(--poe)">
              plain PoE composes 0% on every one of these
            </text>
          </Plot>
          <Legend items={[
            { color: 'var(--lora)', label: 'one held-out pair' },
            { color: 'var(--poe)', label: 'plain PoE' },
          ]} />
          <Stat items={[
            { label: 'adapter composes', value: pct(t.meanCompose), tone: 'pass' },
            { label: 'inside the subspace', value: pct(t.meanGeometry), tone: 'fail' },
            { label: 'held-out pairs', value: String(pairs.length) },
            { label: 'eval step', value: t.evalStep },
          ]} />
          <div className="note bad">
            <b>The geometry is wrong about transfer.</b> Almost none of an unseen
            pair's correction lies in the subspace fitted to the training pairs
            ({pct(t.meanGeometry)}), yet the same adapter takes those pairs from
            total failure to {pct(t.meanCompose)}. Any sentence built on "shared
            subspace" wording has to be rewritten to this bounded form.
          </div>
          <Src keys={['spectrum', 'f6_transfer', 'f6_query']} />
        </div>

        <div className="grow panel">
          <h3>Per pair</h3>
          <table>
            <thead>
              <tr><th>pair</th><th className="num">adapter composes</th>
                  <th className="num">inside subspace</th></tr>
            </thead>
            <tbody>
              {pairs.map((p) => (
                <tr key={p.pair}>
                  <td>{p.pair.replace(/_/g, ' ')}</td>
                  <td className="num" style={{ color: 'var(--lora)' }}>{pct(p.composeRate)}</td>
                  <td className="num" style={{ color: 'var(--control-b)' }}>{pct(p.geometryK64)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h3>Which arms this claim has</h3>
        <Arms
          poe="0% compose on every held-out pair, 8 seeds"
          mono="the teacher is the target the adapter is trained towards"
          lora={`one adapter, ${pct(t.meanCompose)} on ${pairs.length} unseen pairs at step ${t.evalStep}`}
        />
      </div>

      <Cannot>
        why the adapter transfers. It rules out one explanation, that the
        corrections share a low-dimensional subspace, and says nothing about what
        the real one is.
      </Cannot>
    </>
  )
}

// --------------------------------------------------------------- owed page

export function Owed() {
  return (
    <>
      <div className="panel">
        <h3>What is owed, cheapest first</h3>
        {D.gaps.map((g) => (
          <div key={g.id} style={{ marginBottom: 22 }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>
              {g.id} · closes {g.closes}
            </div>
            <div style={{ fontSize: 13.5, marginBottom: 6 }}>{g.what}</div>
            <div style={{ fontSize: 13, color: 'var(--ink-dim)', marginBottom: 8 }}>{g.why}</div>
            <pre>{g.command}</pre>
            <div className="src">lands in {g.output} · cost: {g.cost}</div>
          </div>
        ))}
      </div>

      <div className="panel">
        <h3>Where this page disagrees with the review file</h3>
        <table>
          <thead>
            <tr><th>claim</th><th>the review says</th><th>the data says</th><th>effect</th></tr>
          </thead>
          <tbody>
            {D.discrepancies.map((d, i) => (
              <tr key={i}>
                <td>{d.claim}</td>
                <td>{d.says}</td>
                <td style={{ color: 'var(--warn)' }}>{d.found}</td>
                <td>{d.effect}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note">
          Nothing here has been written back to the review file. The page reads
          the paperwork and rewrites none of it.
        </div>
      </div>
    </>
  )
}
