// Shared pieces: the plot frame, the provenance line, the picture strip.
// Nothing here holds a number; every value arrives as a prop from the store,
// which reads it from the one generated data file.
import { scaleLinear } from 'd3-scale'
import { line as d3line, area as d3area } from 'd3-shape'
import type { ReactNode } from 'react'
import { useShallow } from 'zustand/react/shallow'
import { D, useScene } from './store'
import type { Quote } from './types'

export const W = 460
export const H = 230
export const M = { t: 12, r: 14, b: 34, l: 46 }

export type Scale = ReturnType<typeof scaleLinear<number, number>>

export function Plot({
  x, y, xLabel, yLabel, children, width = W, height = H,
}: {
  x: Scale; y: Scale; xLabel: string; yLabel: string
  children: ReactNode; width?: number; height?: number
}) {
  const xt = x.ticks(6)
  const yt = y.ticks(5)
  return (
    <svg width={width} height={height} role="img">
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={M.l} x2={width - M.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <text x={M.l - 7} y={y(t) + 4} textAnchor="end" fontSize="10"
                fill="var(--ink-faint)" fontFamily="var(--mono-font)">
            {fmtTick(t)}
          </text>
        </g>
      ))}
      {xt.map((t) => (
        <text key={`x${t}`} x={x(t)} y={height - M.b + 15} textAnchor="middle"
              fontSize="10" fill="var(--ink-faint)" fontFamily="var(--mono-font)">
          {fmtTick(t)}
        </text>
      ))}
      {children}
      <text x={(M.l + width - M.r) / 2} y={height - 3} textAnchor="middle"
            fontSize="11" fill="var(--ink-dim)">{xLabel}</text>
      <text x={11} y={(M.t + height - M.b) / 2} textAnchor="middle" fontSize="11"
            fill="var(--ink-dim)" transform={`rotate(-90 11 ${(M.t + height - M.b) / 2})`}>
        {yLabel}
      </text>
    </svg>
  )
}

function fmtTick(t: number) {
  if (Math.abs(t) >= 1000) return t.toExponential(0)
  if (Number.isInteger(t)) return String(t)
  return t.toFixed(Math.abs(t) < 0.1 ? 3 : 2)
}

export function Line({ xs, ys, x, y, color, width = 1.8, dash }: {
  xs: number[]; ys: number[]; x: Scale; y: Scale
  color: string; width?: number; dash?: string
}) {
  const gen = d3line<number>().x((_, i) => x(xs[i])).y((_, i) => y(ys[i]))
  return <path d={gen(ys) ?? ''} fill="none" stroke={color}
               strokeWidth={width} strokeDasharray={dash} />
}

export function Band({ xs, lo, hi, x, y, color }: {
  xs: number[]; lo: number[]; hi: number[]; x: Scale; y: Scale; color: string
}) {
  const gen = d3area<number>()
    .x((_, i) => x(xs[i]))
    .y0((_, i) => y(lo[i]))
    .y1((_, i) => y(hi[i]))
  return <path d={gen(lo) ?? ''} fill={color} opacity={0.18} />
}

export function Marker({ x, at, height = H }: { x: Scale; at: number; height?: number }) {
  return <line x1={x(at)} x2={x(at)} y1={M.t} y2={height - M.b}
               stroke="var(--ink)" strokeWidth={1} strokeDasharray="3 3" opacity={0.6} />
}

/** Where a number came from. Untagged numbers do not ship. */
export function Src({ keys }: { keys: string[] }) {
  return (
    <div className="src">
      {keys.map((k) => {
        const s = D.meta.builtFrom[k]
        if (!s) return <div key={k}>missing stamp: {k}</div>
        return <div key={k}>{s.path}  ·  {s.mtime}</div>
      })}
    </div>
  )
}

/** The review file's own words, with its markdown punctuation taken off so the
 *  sentence reads. The wording itself is never changed. */
export function plainText(md: string) {
  return md
    .replace(/^\s*- \[[ x]\]\s*/gm, '')   // checkbox
    .replace(/^\s*[-*]\s+/gm, '')          // bullet
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') // link, keep the text
    .replace(/\*\*(.+?)\*\*/g, '$1')       // bold
    .replace(/`([^`]+)`/g, '$1')           // code ticks
    .trim()
}

export function Quoted({ q, label }: { q: Quote; label?: string }) {
  const rel = q.path.split('/').slice(-2).join('/')
  return (
    <div>
      <div className="quote">{plainText(q.body.join('\n'))}</div>
      <div className="src">{label ?? rel} : {q.lines[0]}-{q.lines[1]}</div>
    </div>
  )
}

/** A picture with its caption. Click opens it full size. */
export function Frame({ src, cap, kind, wide }: {
  src: string | null; cap: string; kind?: string; wide?: boolean
}) {
  const set = useScene((s) => s.set)
  if (!src) {
    return (
      <div className={`frame ${kind ?? ''}`}>
        <div className="slot" style={{ width: wide ? 190 : 128, height: wide ? 190 : 128,
                                       display: 'grid', placeItems: 'center' }}>
          no frame
        </div>
        <div className="cap">{cap}</div>
      </div>
    )
  }
  return (
    <div className={`frame ${kind ?? ''} ${wide ? 'wide' : ''}`}>
      <img src={src} alt={cap} loading="lazy" onClick={() => set({ lightbox: src })} />
      <div className="cap">{cap}</div>
    </div>
  )
}

export function Lightbox() {
  const { lightbox, set } = useScene(useShallow((s) => ({ lightbox: s.lightbox, set: s.set })))
  if (!lightbox) return null
  return (
    <div className="lightbox" onClick={() => set({ lightbox: null })}>
      <img src={lightbox} alt="full size" />
    </div>
  )
}

export function Stat({ items }: { items: { label: string; value: string; tone?: string }[] }) {
  return (
    <div className="stat">
      {items.map((it) => (
        <div key={it.label}>
          <div className={`v ${it.tone ?? ''}`}>{it.value}</div>
          <div className="l">{it.label}</div>
        </div>
      ))}
    </div>
  )
}

export function Cannot({ children }: { children: ReactNode }) {
  return <div className="cannot">This view cannot tell you: {children}</div>
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}><i style={{ background: i.color }} />{i.label}</span>
      ))}
    </div>
  )
}
