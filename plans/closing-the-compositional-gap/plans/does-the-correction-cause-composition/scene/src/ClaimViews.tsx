// One view per claim, each getting the treatment its type calls for. Claims of the same
// type get the same treatment, so none is thinner than its neighbour by accident.
import { useMemo } from 'react'
import { D, ROW_COLOR, useScene, ALL_PAIRS } from './store'
import { cellsFor, fmt, monotoneBySeed, oneCell, pct, prettyPair, summarize, selectCells } from './compute'
import { CellThumb, GapSlot, Limits, MiniCell, Prov, Push, Quoted, Tag } from './parts'
import { DoseScene } from './DoseScene'
import type { Claim } from './types'

const claim = (id: string) => D.claims.find((c) => c.id === id)!

function Head({ c, lede }: { c: Claim; lede?: string }) {
  return (
    <>
      <h2>{c.question.replace(/\*\*/g, '')}</h2>
      {lede && <p className="lede">{lede}</p>}
    </>
  )
}

/* C2: the same cells behind every strength ---------------------------------- */
function EqualCells() {
  const c = claim('C2')
  const rows = useMemo(() => summarize(D, { pairs: ALL_PAIRS, seeds: [9, 10, 11, 12] }), [])
  const sup = D.supersededTable
  return (
    <div className="panel">
      <Head c={c} lede="A curve whose ends hold more cells than its middle is not one curve. This counts the cells behind every point." />
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>row</th>
              {D.lambdas.map((l) => (
                <th key={l}>λ={l}</th>
              ))}
              <th>equal</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.row}>
                <td>{D.rows.find((x) => x.key === r.row)!.label}</td>
                {r.cellsPerPoint.map((n, i) => (
                  <td className="num" key={i}>{n}</td>
                ))}
                <td className="num" style={{ color: r.equalCells ? 'var(--ok)' : 'var(--flag)' }}>
                  {r.equalCells ? 'yes' : 'no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Prov>
        <b>counted</b> in this page from the cell records. The restriction that made them equal is{' '}
        {D.constants.sweepSeeds.name} = {D.constants.sweepSeeds.raw} at{' '}
        {D.constants.sweepSeeds.path}:{D.constants.sweepSeeds.line}, so a later change to it shows
        up in a diff rather than in a shell history. <Tag kind="derived" />
      </Prov>

      <h3>The read before the size floor</h3>
      <p className="lede" style={{ marginBottom: 8 }}>{sup.why}</p>
      <div className="scroll-x">
        <table>
          <tbody>
            {sup.rows
              .filter((r) => !r[0].startsWith('---'))
              .map((r, i) => (
                <tr key={i} className={i === 0 ? undefined : 'superseded'}>
                  {r.map((cell, j) => (
                    <td key={j} className={j ? 'num' : undefined}>{cell.replace(/\*\*/g, '')}</td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <Prov>
        <b>quoted</b> from {sup.path}:{sup.lines[0]}-{sup.lines[1]}. <Tag kind="quoted" /> Struck
        through because the results file that produced these figures was overwritten by the
        re-score, so this page cannot recompute or check them.
      </Prov>
      <Push
        items={[
          {
            q: 'Did pinning the seeds rescue the result?',
            a: (
              (() => {
                const now = pct(rows.find((r) => r.row === 'oracle')!.points.at(-1)!.rate)
                const before = sup.rows
                  .find((r) => r[0].startsWith('real'))
                  ?.at(-2)
                  ?.replace(/\*\*/g, '')
                return (
                  <>
                    No, and that is the useful part. The endpoint is {now}
                    {now === before ? ', unchanged from before the floor' : `, against ${before} before`}
                    , and the areas barely moved, so the stray cells were not inflating anything and
                    the paper owes no methods sentence about them.
                  </>
                )
              })()
            ),
          },
        ]}
      />
      <Limits>whether the excluded cells were wrong, only that they were not this sweep's.</Limits>
    </div>
  )
}

/* C3: the size floor -------------------------------------------------------- */
function SizeFloor() {
  const c = claim('C3')
  const boxes = D.figures.stripBoxes
  return (
    <div className="panel">
      <Head c={c} lede="The bar that decides whether a detection counts as an animal. Chosen by looking at the boxes. It lives in source, so moving it shows up in a diff." />
      <div className="readouts" style={{ gridTemplateColumns: 'repeat(2, minmax(0,1fr))' }}>
        <div className="readout">
          <div className="name">the size floor, live from source</div>
          <div className="big">{D.constants.minBoxFraction.raw}</div>
          <div className="sub">
            {D.constants.minBoxFraction.path}:{D.constants.minBoxFraction.line}
          </div>
        </div>
        <div className="readout">
          <div className="name">what it means</div>
          <div className="big" style={{ fontSize: 17, lineHeight: 1.35 }}>
            a detection must span a quarter of the image's longer side
          </div>
          <div className="sub">it can only remove detections, so it can only lower rates</div>
        </div>
      </div>
      {boxes && (
        <>
          <h3>The picture the bar was set from</h3>
          <a href={boxes.publicPath} target="_blank" rel="noreferrer">
            <img src={boxes.publicPath} alt="annotated boxes strip" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--rule)' }} />
          </a>
          <Prov>
            <b>image</b> {boxes.path}, written {boxes.mtime}. Yellow boxes clear the floor, magenta
            ones do not, drawn by the same line the scorer applies. <Tag kind="measured" />
          </Prov>
        </>
      )}
      <h3>What no file holds</h3>
      <Quoted claim={c} />
      <GapSlot claim={c} />
      <Push
        items={[
          {
            q: 'Was the threshold moved to rescue the result?',
            a: <>It cannot have been: a size floor only deletes detections, so it can only push
              compose rates down. It pulled the controls down further than the oracle, which is
              what should happen if the controls' readings were instrument error.</>,
          },
          {
            q: 'Why not tighten confidence instead?',
            a: <>The false box scored 0.60, above the real penguin's 0.54 on the same strip, so no
              confidence cutoff separates them. That comparison is quoted above, not measured here.</>,
          },
        ]}
      />
      <Limits>
        whether 0.25 is right for images that are not 1024px, or for pairs outside this sweep.
      </Limits>
    </div>
  )
}

/* C4 and C5: answered in the review, no artifact ---------------------------- */
function QuotedClaim({ id, lede }: { id: string; lede: string }) {
  const c = claim(id)
  return (
    <div className="panel">
      <Head c={c} lede={lede} />
      <Quoted claim={c} />
      <GapSlot claim={c} />
      {id === 'C4' && (
        <Prov>
          <b>the tests exist</b> at {D.meta.builtFrom.canaries.path}, last changed{' '}
          {D.meta.builtFrom.canaries.mtime}. They assert exact equality against the sampler's own
          saved output, which is stricter than the 1e-5 the plan pre-registered. What is missing is
          a stored record that they passed, not the tests.
        </Prov>
      )}
    </div>
  )
}

/* C6: the smoke cell the eye and the scorer agreed on ----------------------- */
function SmokeCell() {
  const c = claim('C6')
  const set = useScene((s) => s.set)
  const pair = 'a_cat__x__a_dog'
  const seed = 9
  const trio = cellsFor(D, { pair, seed, lam: 1 })
  return (
    <div className="panel">
      <Head c={c} lede="One cell, checked by eye before the sweep was trusted. Each picture carries the scorer's verdict, so you can disagree with it." />
      <div className="cellrow" style={{ gridTemplateColumns: 'repeat(3, minmax(0,1fr))' }}>
        {trio.map((t) => (
          <div key={t.row}>
            <div className="rowname" style={{ marginBottom: 6 }}>
              <span className="swatch" style={{ background: ROW_COLOR[t.row], marginRight: 6 }} />
              {D.rows.find((r) => r.key === t.row)!.label}
            </div>
            <CellThumb cell={t.cell} />
          </div>
        ))}
      </div>
      <Prov>
        <b>cells</b> {prettyPair(pair)}, seed {seed}, λ=1, from {D.meta.imagesRoot}.{' '}
        <Tag kind="measured" /> The verdicts are the scorer's, recomputed nowhere: they are the
        records' own `compose` field.
      </Prov>
      <Push
        items={[
          {
            q: 'One cell is not evidence.',
            a: <>Correct, and it is not offered as any. It is the check that the instrument and the
              eye agree before an hour of scoring was trusted. The evidence is C1.{' '}
              <button className="chip" style={{ display: 'inline-flex', padding: '2px 9px' }}
                onClick={() => set({ claim: 'C1' })}>go to C1</button></>,
          },
        ]}
      />
      <Limits>anything about the other 479 cells.</Limits>
    </div>
  )
}

/* C7: the assembled figure, and what it does not smooth over ---------------- */
function PerPairRanking() {
  const rank = useMemo(() => {
    const pairs = [...new Set(D.cells.map((c) => c.pair))]
    return pairs
      .map((p) => {
        const rows = summarize(D, { pairs: [p], seeds: [9, 10, 11, 12] })
        const o = rows.find((r) => r.row === 'oracle')!
        const controls = rows.filter((r) => r.row !== 'oracle')
        return {
          pair: p,
          points: o.points.map((x) => x.rate),
          auc: o.auc,
          controlAucs: controls.map((x) => x.auc),
          controlsAtZero: controls.every((x) => x.auc === 0),
        }
      })
      .sort((a, b) => b.auc - a.auc)
  }, [])
  const bothZero = rank.filter((r) => r.controlsAtZero)
  const top = rank[0]
  return (
    <>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>pair</th>
              <th>at λ=1</th>
              {D.lambdas.map((l) => <th key={l}>λ={l}</th>)}
              <th>area</th>
              <th>control areas</th>
            </tr>
          </thead>
          <tbody>
            {rank.map((r) => (
              <tr key={r.pair} style={r.pair === top.pair ? { fontWeight: 600 } : undefined}>
                <td>{prettyPair(r.pair)}</td>
                <td><MiniCell cell={oneCell(D, { row: 'oracle', pair: r.pair, seed: 9, lam: 1 })} /></td>
                {r.points.map((p, i) => <td className="num" key={i}>{pct(p)}</td>)}
                <td className="num">{fmt(r.auc)}</td>
                <td className="num" style={{ color: r.controlsAtZero ? 'var(--ok)' : undefined }}>
                  {r.controlAucs.map((a) => fmt(a)).join('  ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Prov>
        <b>ranked</b> in this page, one point per pair over 4 seeds. Each picture is that pair's
        oracle cell at λ=1, seed 9, picked by that rule and not by which one looks best. A thick
        border means the scorer called it composed. <Tag kind="derived" />{' '}
        {prettyPair(top.pair)} is the strongest pair, which is the reason the review gives for
        putting it in the figure, and that part holds.{' '}
        <b style={{ color: 'var(--flag)' }}>
          The review also calls it the only pair whose two controls score exactly 0.000. It is not:{' '}
          {bothZero.length} of {rank.length} pairs do ({bothZero.map((r) => prettyPair(r.pair)).join(', ')}).
        </b>{' '}
        The choice of pair still stands on the area; the "only pair" reason does not.
      </Prov>
    </>
  )
}

function TheStrip() {
  const c = claim('C7')
  const f2 = D.figures.f2
  const f2b = D.figures.f2b
  const perSeed = useMemo(() => monotoneBySeed(D, ['an_elephant__x__a_penguin']), [])
  const threeAnimals = useMemo(
    () =>
      D.cells.filter(
        (x) => x.pair === 'an_elephant__x__a_penguin' && x.seed === 10 && x.nInstances >= 3,
      ),
    [],
  )
  return (
    <div className="panel">
      <Head c={c} lede="F2: cat and dog, seed 9, three rows across five strengths above the curves. Read down a column and the injected vector is what changes, not the size of the nudge." />
      {f2 && (
        <>
          <a href={f2.publicPath} target="_blank" rel="noreferrer">
            <img src={f2.publicPath} alt="F2, dose response with cells" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--rule)' }} />
          </a>
          <Prov>
            <b>figure</b> {f2.path}, assembled {f2.mtime} by{' '}
            {D.meta.builtFrom.figureScript?.path ?? 'the figure script'}. <Tag kind="measured" />{' '}
            This is the file the paper includes, not a redrawing of it.
          </Prov>
        </>
      )}

      <h3>Why this pair carries the figure</h3>
      <PerPairRanking />

      {f2b && (
        <>
          <h3>The second figure, answering "they only look alike"</h3>
          <p className="lede">
An elephant and a penguin share nothing. PoE still fuses them at λ=0, so the failure is not
            the two animals looking alike.
          </p>
          <a href={f2b.publicPath} target="_blank" rel="noreferrer">
            <img src={f2b.publicPath} alt="F2b, the dissimilar pair" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--rule)' }} />
          </a>
          <Prov>
            <b>figure</b> {f2b.path}, assembled {f2b.mtime}. <Tag kind="measured" />
          </Prov>
        </>
      )}

      <h3>What the second figure should not smooth over</h3>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>seed</th>
              <th>at λ=1</th>
              {D.lambdas.map((l) => <th key={l}>λ={l}</th>)}
              <th>rises at every step</th>
            </tr>
          </thead>
          <tbody>
            {perSeed.map((s) => (
              <tr key={s.seed}>
                <td>{s.seed}</td>
                <td>
                  <MiniCell cell={oneCell(D, {
                    row: 'oracle', pair: 'an_elephant__x__a_penguin', seed: s.seed, lam: 1 })} />
                </td>
                {s.points.map((p, i) => <td className="num" key={i}>{pct(p)}</td>)}
                <td className="num" style={{ color: s.monotone ? 'var(--ok)' : 'var(--flag)' }}>
                  {s.monotone ? 'yes' : 'no'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Prov>
        <b>computed</b> per seed for elephant × penguin from the cell records, one cell per point.
        Rising at every step: seeds {perSeed.filter((s) => s.monotone).map((s) => s.seed).join(', ')}.
        Not: seeds {perSeed.filter((s) => !s.monotone).map((s) => s.seed).join(', ') || 'none'}.{' '}
        <Tag kind="derived" />{' '}
        <b style={{ color: 'var(--flag)' }}>
          The review says seeds 9 and 11 of this pair are not monotone. On these records only seed{' '}
          {perSeed.filter((s) => !s.monotone).map((s) => s.seed).join(' and ')} fails: seed 11 goes{' '}
          {perSeed.find((s) => s.seed === 11)?.points.map((p) => pct(p)).join(', ')}, which never
          falls.
        </b>
      </Prov>
      <p style={{ fontSize: 13.5, color: 'var(--ink-2)' }}>
        At λ=1 the panel holds <b>{threeAnimals.length ? threeAnimals[0].nInstances : '?'}</b>{' '}
        animals rather than two, which is where the extra count comes from. Cells scoring three or
        more on this pair and seed: {threeAnimals.length} of {D.lambdas.length * D.rows.length}.
      </p>
      <Push
        items={[
          {
            q: 'Was the pair picked because it flatters the result?',
            a: <>Yes, in the sense that it is the strongest pair in the set, and the table above
              says so plainly rather than leaving a reader to find it. The second figure exists
              because that choice invites the objection, and it uses the pair with nothing in
              common.</>,
          },
          {
            q: 'At λ=0.5 the failure changes character.',
            a: <>It stops fusing and drops the penguin entirely. That is a different failure from
              blending, and it is not visible in the compose rate, which counts both as "not two
              animals".</>,
          },
        ]}
      />
      <Limits>
        how often each kind of failure happens. One strip is one pair, one seed, and the rate on C1
        does not distinguish a fused animal from a missing one.
      </Limits>
    </div>
  )
}

/* C8: the control the pool does not have ------------------------------------ */
function NoDoNoHarm() {
  const c = claim('C8')
  const zero = useMemo(
    () =>
      selectCells(D.cells, { pairs: ['an_elephant__x__a_penguin'], seeds: [9, 10, 11, 12] }).filter(
        (x) => x.lam === 0 && x.row === 'oracle',
      ),
    [],
  )
  const composed = zero.filter((z) => z.compose).length
  return (
    <div className="panel">
      <Head c={c} lede="The pool lists elephant × penguin as the pair that composes without help. It does not. So the do-no-harm check does not exist." />
      <div className="readouts" style={{ gridTemplateColumns: 'repeat(2, minmax(0,1fr))' }}>
        <div className="readout">
          <div className="name">composes at λ=0, with nothing injected</div>
          <div className="big" style={{ color: 'var(--flag)' }}>{composed} of {zero.length}</div>
          <div className="sub">seeds {zero.map((z) => z.seed).join(', ')}</div>
        </div>
        <div className="readout">
          <div className="name">the pool's assumption</div>
          <div className="big" style={{ fontSize: 17, lineHeight: 1.35 }}>
            listed as the compose-by-default control
          </div>
          <div className="sub">{D.meta.builtFrom.pairPool.path}</div>
        </div>
      </div>
      <div className="cellrow" style={{ gridTemplateColumns: 'repeat(4, minmax(0,1fr))', marginTop: 14 }}>
        {zero.map((z) => (
          <CellThumb key={z.seed} cell={z} label={`seed ${z.seed} at lambda 0`} />
        ))}
      </div>
      <Prov>
        <b>counted and shown</b> from the cell records and the images behind them.{' '}
        <Tag kind="measured" /> The scorer is right and the pool's assumption is wrong: all four are
        single fused creatures.
      </Prov>
      <Push
        items={[
          {
            q: 'So what is owed?',
            a: <>A limitations sentence: this pool has no working do-no-harm control, so nothing
              here shows the correction leaves an already-composing pair alone.</>,
          },
        ]}
      />
      <Limits>whether some other pair in the pool would serve as the missing control.</Limits>
    </div>
  )
}

/* C9: where the output landed ------------------------------------------------ */
function WhereItLanded() {
  const c = claim('C9')
  const e = D.environment
  const gb = (e.outputBytes / 1024 ** 3).toFixed(1)
  return (
    <div className="panel">
      <Head c={c} lede="A status card, not a result. The one open task on this plan. The disk guard read a filesystem the run never wrote to." />
      <div className="readouts" style={{ gridTemplateColumns: 'repeat(2, minmax(0,1fr))' }}>
        <div className="readout">
          <div className="name">cells on disk, measured at build time</div>
          <div className="big" style={{ color: 'var(--flag)' }}>{gb} GiB</div>
          <div className="sub">on {e.outputFilesystem}, which is not where the plan said</div>
        </div>
        <div className="readout">
          <div className="name">what the guard checked</div>
          <div className="big" style={{ fontSize: 17, lineHeight: 1.35 }}>{e.guardFilesystem}</div>
          <div className="sub">a filesystem the run never wrote to</div>
        </div>
      </div>
      <h3>The two lines</h3>
      <code className="cmd">
        {e.scriptWrites.path}:{e.scriptWrites.line}   {e.scriptWrites.text}
        {'\n'}
        {e.scriptGuardChecks.path}:{e.scriptGuardChecks.line}   {e.scriptGuardChecks.text}
      </code>
      <Prov>
        <b>size</b> measured with du at build time, <b>lines</b> read from the sweep script.{' '}
        <Tag kind="measured" /> Open task: {e.planTask.path}:{e.planTask.line}
      </Prov>
      <Push
        items={[
          {
            q: 'Does this invalidate the result?',
            a: <>No. It is where the bytes sit, not what they say. It does mean the guard would not
              have caught a full disk, so the next long run is the risk, not this one.</>,
          },
        ]}
      />
      <Limits>whether /datasets had room at the time; the guard's reading was about the wrong disk.</Limits>
    </div>
  )
}

export function ClaimView({ id }: { id: string }) {
  switch (id) {
    case 'C1': return <DoseScene />
    case 'C2': return <EqualCells />
    case 'C3': return <SizeFloor />
    case 'C4':
      return <QuotedClaim id="C4" lede="Does our own code move the baseline everything else is measured against? Eight tests assert exact equality, each shown to fail against a deliberately broken sampler." />
    case 'C5':
      return <QuotedClaim id="C5" lede="The controls differ from the oracle in direction, not size. This is what makes them controls. It is the one claim with nothing on disk behind it." />
    case 'C6': return <SmokeCell />
    case 'C7': return <TheStrip />
    case 'C8': return <NoDoNoHarm />
    case 'C9': return <WhereItLanded />
    default: return null
  }
}

export const claimSummary = () =>
  D.claims.map((c) => ({
    id: c.id,
    state: c.state,
    flagged: c.mark === '❌',
    label: c.id,
    title: c.question.replace(/\*\*/g, ''),
  }))

export { fmt }
