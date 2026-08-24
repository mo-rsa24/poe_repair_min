// The paper's headline claim, driveable. Curve, readouts and cells all read one
// scrubber. Every rate is recomputed here from the 480 records and checked against the
// values the results file carries.
import { useMemo } from 'react'
import { line } from 'd3-shape'
import { scaleLinear } from 'd3-scale'
import { D, ROW_COLOR, useScene, ALL_PAIRS, ALL_SEEDS } from './store'
import { cellsFor, fmt, oneCell, pct, prettyPair, summarize, thumbUrl } from './compute'
import { CellThumb, Limits, Prov, Push, Tag } from './parts'
import type { RowKey } from './types'

const W = 560
const H = 330
const M = { t: 28, r: 30, b: 44, l: 60 }   // room for the cells riding the end points

export function DoseCurves() {
  const lam = useScene((s) => s.lam)
  const row = useScene((s) => s.row)
  const pairs = useScene((s) => s.pairs)
  const seeds = useScene((s) => s.seeds)
  const set = useScene((s) => s.set)
  const rows = useMemo(() => summarize(D, { pairs, seeds }), [pairs, seeds])

  const x = scaleLinear().domain([0, 1]).range([M.l, W - M.r])
  const y = scaleLinear().domain([0, 1]).range([H - M.b, M.t])
  const mk = line<{ l: number; v: number }>()
    .x((d) => x(d.l))
    .y((d) => y(d.v))

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label="Compose rate against correction strength, three rows">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--rule)" />
          <text x={M.l - 9} y={y(t) + 4} textAnchor="end" fontSize="10.5"
            fontFamily="var(--mono)" fill="var(--ink-3)">
            {t * 100}%
          </text>
        </g>
      ))}
      {D.lambdas.map((l) => (
        <text key={l} x={x(l)} y={H - M.b + 17} textAnchor="middle" fontSize="10.5"
          fontFamily="var(--mono)" fill={l === lam ? 'var(--ink)' : 'var(--ink-3)'}>
          {l}
        </text>
      ))}
      <text x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle" fontSize="11"
        fontFamily="var(--mono)" fill="var(--ink-3)">
        λ, how much of the correction is added back
      </text>

      {/* the scrubbed strength, drawn behind the curves */}
      <line x1={x(lam)} x2={x(lam)} y1={M.t} y2={H - M.b} stroke="var(--ink-3)"
        strokeDasharray="3 3" />

      {rows.map((r) => {
        const pts = D.lambdas.map((l, i) => ({ l, v: r.points[i].rate }))
        const isOracle = r.row === 'oracle'
        return (
          <g key={r.row}>
            <path d={mk(pts.filter((p) => !Number.isNaN(p.v))) ?? ''} fill="none"
              stroke={ROW_COLOR[r.row]} strokeWidth={isOracle ? 2.4 : 1.5}
              strokeDasharray={isOracle ? undefined : r.row === 'random' ? '5 4' : '2 3'} />
            {pts.map((p) =>
              Number.isNaN(p.v) ? null : row === r.row ? (
                // The picture rides the point: this row's actual cell for the shown pair
                // and seed, bordered in the row's colour, enlarged at the scrubbed value.
                <CellMarker key={p.l} row={r.row} lam={p.l} cx={x(p.l)} cy={y(p.v)}
                  size={p.l === lam ? 46 : 30} />
              ) : (
                <circle key={p.l} cx={x(p.l)} cy={y(p.v)} r={p.l === lam ? 5 : 3}
                  fill={p.l === lam ? ROW_COLOR[r.row] : 'var(--panel)'}
                  stroke={ROW_COLOR[r.row]} strokeWidth="1.6"
                  style={{ cursor: 'pointer' }} onClick={() => set({ lam: p.l, row: r.row })} />
              ),
            )}
          </g>
        )
      })}
    </svg>
  )
}

/** A cell riding a point on the curve. Border colour is the row, border weight says
 *  whether the scorer called it composed, so the picture and the verdict arrive together. */
function CellMarker({ row, lam, cx, cy, size }: {
  row: RowKey; lam: number; cx: number; cy: number; size: number
}) {
  const pair = useScene((s) => s.pair)
  const seed = useScene((s) => s.seed)
  const set = useScene((s) => s.set)
  const cell = oneCell(D, { row, pair, seed, lam })
  if (!cell) return <circle cx={cx} cy={cy} r={3} fill="var(--ink-3)" />
  const id = `clip-${row}-${lam}`.replace(/\./g, '_')
  const s = size
  return (
    <g style={{ cursor: 'pointer' }} onClick={() => set({ lam, row })}>
      <clipPath id={id}>
        <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} rx={5} />
      </clipPath>
      <image href={thumbUrl(cell.image.relPath)} x={cx - s / 2} y={cy - s / 2} width={s}
        height={s} preserveAspectRatio="xMidYMid slice" clipPath={`url(#${id})`} />
      <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} rx={5} fill="none"
        stroke={ROW_COLOR[row]} strokeWidth={cell.compose ? 3 : 1.2} />
      {cell.compose && (
        <circle cx={cx + s / 2 - 4} cy={cy - s / 2 + 4} r={3.5} fill={ROW_COLOR[row]} />
      )}
    </g>
  )
}

export function RowPicker() {
  const row = useScene((s) => s.row)
  const set = useScene((s) => s.set)
  return (
    <div className="field">
      <label>cells shown on the curve</label>
      <div className="seg">
        {D.rows.map((r) => (
          <button key={r.key} aria-pressed={r.key === row} onClick={() => set({ row: r.key })}
            style={{ color: r.key === row ? ROW_COLOR[r.key] : undefined }}>
            {r.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function Readouts() {
  const lam = useScene((s) => s.lam)
  const pairs = useScene((s) => s.pairs)
  const seeds = useScene((s) => s.seeds)
  const rows = useMemo(() => summarize(D, { pairs, seeds }), [pairs, seeds])
  const i = D.lambdas.indexOf(lam)
  return (
    <div className="readouts">
      {rows.map((r) => {
        const p = r.points[i]
        const label = D.rows.find((x) => x.key === r.row)!.label
        return (
          <div className="readout" key={r.row}>
            <div className="name">
              <span className="swatch" style={{ background: ROW_COLOR[r.row] }} />
              {label}
            </div>
            <div className="big" style={{ color: r.row === 'oracle' ? 'var(--oracle)' : undefined }}>
              {pct(p.rate)}
            </div>
            <div className="sub">
              {p.k} of {p.n} cells · area {fmt(r.auc)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function Scrubber() {
  const lam = useScene((s) => s.lam)
  const set = useScene((s) => s.set)
  const i = D.lambdas.indexOf(lam)
  return (
    <div className="field">
      <label htmlFor="lam">strength λ = {lam}</label>
      <input id="lam" type="range" min={0} max={D.lambdas.length - 1} step={1} value={i}
        onChange={(e) => set({ lam: D.lambdas[Number(e.target.value)] })} />
    </div>
  )
}

export function PairSeedPicker() {
  const pair = useScene((s) => s.pair)
  const seed = useScene((s) => s.seed)
  const set = useScene((s) => s.set)
  return (
    <>
      <div className="field">
        <label htmlFor="pair">pair shown below</label>
        <select id="pair" value={pair} onChange={(e) => set({ pair: e.target.value })}
          style={{ padding: '6px 8px', fontFamily: 'var(--mono)', fontSize: 12 }}>
          {ALL_PAIRS.map((p) => (
            <option key={p} value={p}>{prettyPair(p)}</option>
          ))}
        </select>
      </div>
      <div className="field">
        <label>seed</label>
        <div className="seg">
          {ALL_SEEDS.map((s) => (
            <button key={s} aria-pressed={s === seed} onClick={() => set({ seed: s })}>{s}</button>
          ))}
        </div>
      </div>
    </>
  )
}

/** The three rows by five strengths, for one pair and seed. The qualitative half. */
export function CellGrid() {
  const pair = useScene((s) => s.pair)
  const seed = useScene((s) => s.seed)
  const lam = useScene((s) => s.lam)
  return (
    <div>
      <div className="lamhead">
        <div />
        {D.lambdas.map((l) => (
          <div key={l} data-current={l === lam}>λ={l}</div>
        ))}
      </div>
      {D.rows.map((r) => (
        <div className="cellrow" key={r.key}>
          <div className="rowname">
            <span className="swatch" style={{ background: ROW_COLOR[r.key], marginRight: 6 }} />
            {r.label}
          </div>
          {D.lambdas.map((l) => {
            const cell = cellsFor(D, { pair, seed, lam: l }).find((c) => c.row === r.key)?.cell
            return <CellThumb key={l} cell={cell} current={l === lam} />
          })}
        </div>
      ))}
    </div>
  )
}

export function AgreementNote() {
  const pairs = useScene((s) => s.pairs)
  const seeds = useScene((s) => s.seeds)
  const rows = useMemo(() => summarize(D, { pairs, seeds }), [pairs, seeds])
  const whole = pairs.length === ALL_PAIRS.length && seeds.length === ALL_SEEDS.length
  const agree = rows.every((r) => r.agreesWithFile)
  return (
    <Prov>
      <b>rates and areas</b> recomputed in this page from {D.cells.length} records in{' '}
      {D.meta.builtFrom.curves.path}, scorer {D.fileSummary.scorer}, built{' '}
      {D.meta.builtFrom.curves.mtime}.{' '}
      {whole ? (
        agree ? (
          <>Every value matches the results file exactly, including the areas under the curves.</>
        ) : (
          <b style={{ color: 'var(--flag)' }}>
            These values do NOT match the results file. The page and the file disagree; trust
            neither until that is explained.
          </b>
        )
      ) : (
        <>
          A subset of pairs or seeds is selected, so these are not the sweep's own numbers and no
          comparison against the results file is made.
        </>
      )}
    </Prov>
  )
}

export function DoseScene() {
  const claim = D.claims.find((c) => c.id === 'C1')!
  const pairs = useScene((s) => s.pairs)
  const seeds = useScene((s) => s.seeds)
  const pair = useScene((s) => s.pair)
  const seed = useScene((s) => s.seed)
  const rows = useMemo(() => summarize(D, { pairs, seeds }), [pairs, seeds])
  const oracle = rows.find((r) => r.row === 'oracle')!
  const controls = rows.filter((r) => r.row !== 'oracle')
  const bestControl = Math.max(...controls.map((c) => c.auc))
  const worstControlRise = Math.max(...controls.map((c) => c.rise))

  return (
    <>
      <div className="panel">
        <h2>{claim.question}</h2>
        <p className="lede">
The only question here whose failure moves the plan. Drag λ: the curve, the three rates and
          the cells all move together. The pictures on the curve are the real cells at those points.
        </p>
        <div className="controls" style={{ marginBottom: 14 }}>
          <Scrubber />
          <RowPicker />
          <PairSeedPicker />
        </div>
        <div className="grid2">
          <div>
            <DoseCurves />
          </div>
          <div>
            <Readouts />
            <table style={{ marginTop: 10 }}>
              <tbody>
                <tr>
                  <td>oracle area against the better control</td>
                  <td className="num">{fmt(oracle.auc / bestControl, 1)}×</td>
                </tr>
                <tr>
                  <td>oracle rise, λ=0 to λ=1</td>
                  <td className="num">{Math.round(oracle.rise * 100)} points</td>
                </tr>
                <tr>
                  <td>best control rise</td>
                  <td className="num">{Math.round(worstControlRise * 100)} points</td>
                </tr>
                <tr>
                  <td>cells behind every point</td>
                  <td className="num">
                    {oracle.equalCells ? oracle.cellsPerPoint[0] : oracle.cellsPerPoint.join(' / ')}
                    {oracle.equalCells ? '' : '  (unequal)'}
                  </td>
                </tr>
              </tbody>
            </table>
            <Tag kind="derived" />{' '}
            <span style={{ fontSize: 12.5, color: 'var(--ink-3)' }}>
              every figure in this panel, from the cell records
            </span>
          </div>
        </div>
        <AgreementNote />
      </div>

      <div className="panel">
        <h2>The cells behind the rate</h2>
        <p className="lede">
          {prettyPair(pair)}, seed {seed}. Three rows, five strengths. The badge is the scorer's
          count, not an eye judgement. Click for full size.
        </p>
        <CellGrid />
        <Push
          items={[
            {
              q: 'Why do the three rows read the same at λ=0?',
              a: (
                <>
                  Because nothing is injected there, so one image is scored once per row. The cells
                  marked "shared row" are literally the same file. 480 records come from 416
                  images for exactly this reason.
                </>
              ),
            },
            {
              q: 'Is the rate per cell or per seed?',
              a: (
                <>
                  Per cell. Each point pools {oracle.cellsPerPoint[0] ?? 0} cells, 8 pairs by 4
                  seeds. The seed buttons split it. C7 computes which seeds rise at every step and
                  which do not.
                </>
              ),
            },
            {
              q: 'Could the controls be flat because they inject less?',
              a: (
                <>
                  That is claim C5, and it is the one this page cannot check: the review answers it,
                  but no file in the repo holds the numbers.
                </>
              ),
            },
          ]}
        />
        <Limits>
          which cells failed and why. A rate is a count of verdicts; the picture beside it is one
          cell out of {oracle.cellsPerPoint[0] ?? 0}. It also cannot tell you anything about pairs
          the sweep did not run.
        </Limits>
      </div>
    </>
  )
}

export function rowLabel(k: RowKey) {
  return D.rows.find((r) => r.key === k)!.label
}
