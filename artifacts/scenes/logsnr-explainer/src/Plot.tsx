/**
 * One bespoke SVG plot. Mafs is not used here: every panel is a labelled
 * scientific axis pair rather than a coordinate plane, which is the case the
 * stack notes hand to SVG plus d3 scales.
 */
import { scaleLinear } from 'd3-scale'
import { line as d3line, area as d3area } from 'd3-shape'
import { fmt } from './math'

export type Series = {
  points: [number, number][]
  color: string
  width?: number
  dash?: string
  opacity?: number
  dots?: number // radius; 0 or undefined draws no dots
}

export type Band = { points: [number, number, number][]; color: string; opacity?: number }

export type Marker = { x: number; y?: number; color: string; label?: string }

/** A label placed in data coordinates, on the thing it names. Preferred over
 *  a legend: a legend charges the reader a lookup. */
export type InlineLabel = { x: number; y: number; text: string; color: string }

export type PlotProps = {
  width?: number
  height?: number
  xDomain: [number, number]
  yDomain: [number, number]
  xLabel: string
  yLabel: string
  xColor?: string
  series: Series[]
  bands?: Band[]
  markers?: Marker[]
  inlineLabels?: InlineLabel[]
  hlines?: { y: number; color: string; dash?: string }[]
  yTickFormat?: (v: number) => string
  xTickFormat?: (v: number) => string
}

export function Plot({
  width = 380,
  height = 210,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  xColor = 'var(--ink-soft)',
  series,
  bands = [],
  markers = [],
  inlineLabels = [],
  hlines = [],
  yTickFormat = (v) => fmt(v, 2),
  xTickFormat = (v) => fmt(v, 1),
}: PlotProps) {
  const m = { top: 12, right: 14, bottom: 38, left: 46 }
  const iw = width - m.left - m.right
  const ih = height - m.top - m.bottom
  const x = scaleLinear().domain(xDomain).range([0, iw])
  const y = scaleLinear().domain(yDomain).range([ih, 0])

  const path = d3line<[number, number]>()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
  const areaPath = d3area<[number, number, number]>()
    .x((d) => x(d[0]))
    .y0((d) => y(d[1]))
    .y1((d) => y(d[2]))

  return (
    <svg width={width} height={height} className="plot" role="img">
      <g transform={`translate(${m.left},${m.top})`}>
        {y.ticks(4).map((v) => (
          <g key={`y${v}`}>
            <line x1={0} x2={iw} y1={y(v)} y2={y(v)} className="grid" />
            <text x={-8} y={y(v)} className="tick" textAnchor="end" dominantBaseline="middle">
              {yTickFormat(v)}
            </text>
          </g>
        ))}
        {x.ticks(5).map((v) => (
          <text key={`x${v}`} x={x(v)} y={ih + 16} className="tick" textAnchor="middle">
            {xTickFormat(v)}
          </text>
        ))}

        {bands.map((b, i) => (
          <path
            key={`b${i}`}
            d={areaPath(b.points) ?? undefined}
            fill={b.color}
            opacity={b.opacity ?? 0.18}
          />
        ))}

        {hlines.map((h, i) => (
          <line
            key={`h${i}`}
            x1={0}
            x2={iw}
            y1={y(h.y)}
            y2={y(h.y)}
            stroke={h.color}
            strokeDasharray={h.dash ?? '3 3'}
            strokeWidth={1}
          />
        ))}

        {series.map((s, i) => (
          <g key={`s${i}`} opacity={s.opacity ?? 1}>
            <path
              d={path(s.points) ?? undefined}
              fill="none"
              stroke={s.color}
              strokeWidth={s.width ?? 1.6}
              strokeDasharray={s.dash}
              strokeLinejoin="round"
            />
            {s.dots
              ? s.points.map((p, j) => (
                  <circle key={j} cx={x(p[0])} cy={y(p[1])} r={s.dots} fill={s.color} />
                ))
              : null}
          </g>
        ))}

        {markers.map((mk, i) => (
          <g key={`m${i}`}>
            <line
              x1={x(mk.x)}
              x2={x(mk.x)}
              y1={0}
              y2={ih}
              stroke={mk.color}
              strokeWidth={1.2}
              strokeDasharray="4 3"
            />
            {mk.y !== undefined ? (
              <circle cx={x(mk.x)} cy={y(mk.y)} r={4.5} fill={mk.color} />
            ) : null}
            {mk.label ? (
              <text x={x(mk.x) + 5} y={12} className="marker-label" fill={mk.color}>
                {mk.label}
              </text>
            ) : null}
          </g>
        ))}

        {inlineLabels.map((l, i) => (
          <text key={`il${i}`} x={x(l.x)} y={y(l.y)} className="inline-label" fill={l.color}>
            {l.text}
          </text>
        ))}

        <line x1={0} x2={iw} y1={ih} y2={ih} className="axis" />
        <line x1={0} x2={0} y1={0} y2={ih} className="axis" />
        <text x={iw / 2} y={ih + 33} className="axis-label" textAnchor="middle" fill={xColor}>
          {xLabel}
        </text>
        <text
          transform={`translate(${-34},${ih / 2}) rotate(-90)`}
          className="axis-label"
          textAnchor="middle"
        >
          {yLabel}
        </text>
      </g>
    </svg>
  )
}
