// Small pieces every claim uses: where a number came from, what a reader would push on,
// what the view cannot show, and the slot for a claim with no artifact behind it.
import type { ReactNode } from 'react'
import { D, useScene } from './store'
import { fullUrl, prettyPair, thumbUrl } from './compute'
import type { Cell, Claim } from './types'

export function Prov({ children }: { children: ReactNode }) {
  return <div className="prov">{children}</div>
}

export function Tag({ kind }: { kind: 'measured' | 'quoted' | 'derived' }) {
  const label = kind === 'derived' ? 'computed in this page' : kind
  return <span className={`tag ${kind}`}>{label}</span>
}

export function Push({ items }: { items: { q: string; a: ReactNode }[] }) {
  return (
    <div className="push">
      <h3 style={{ margin: '0 0 8px' }}>What a reader pushes on</h3>
      {items.map((it, i) => (
        <p key={i}>
          <b>{it.q}</b> {it.a}
        </p>
      ))}
    </div>
  )
}

export function Limits({ children }: { children: ReactNode }) {
  return <p className="limits">This view cannot tell you: {children}</p>
}

export function GapSlot({ claim }: { claim: Claim }) {
  const f = claim.fill
  if (!f) return null
  return (
    <div className="gap">
      <h4>Nothing on disk holds this number</h4>
      <p className="lede" style={{ marginBottom: 6 }}>
The review file answers this question and may well be right. No file in the repo carries the
        figures, so this page cannot check them or plot them.
      </p>
      <dl>
        <dt>would show</dt>
        <dd>{f.what}</dd>
        <dt>run</dt>
        <dd>{f.hasCommand ? <code className="cmd">{f.command}</code> : f.command}</dd>
        <dt>lands in</dt>
        <dd>{f.output}</dd>
        <dt>costs</dt>
        <dd>{f.cost}</dd>
      </dl>
    </div>
  )
}

/** The review file's own answer, rewrapped into its paragraphs and quoted verbatim. */
export function Quoted({ claim }: { claim: Claim }) {
  const paras = claim.answer
    .reduce<string[][]>(
      (acc, line) => {
        if (!line.trim()) acc.push([])
        else acc[acc.length - 1].push(line)
        return acc
      },
      [[]],
    )
    .map((p) => p.join(' ').replace(/\*\*/g, '').trim())
    .filter(Boolean)
  return (
    <>
      {paras.map((p, i) => (
        <p className="quote" key={i}>
          {p}
        </p>
      ))}
      <Prov>
        <b>quoted from</b> {claim.source.path}:{claim.source.lines[0]}-{claim.source.lines[1]}
      </Prov>
    </>
  )
}

/** One generated picture, with the scorer's own verdict on it. Click opens full size. */
export function CellThumb({
  cell,
  current = false,
  label,
}: {
  cell: Cell | undefined
  current?: boolean
  label?: string
}) {
  const set = useScene((s) => s.set)
  if (!cell) return <div className="cell" style={{ opacity: 0.35 }} />
  return (
    <button
      className="cell"
      data-current={current}
      onClick={() => set({ lightbox: cell.image.relPath })}
      title={`${prettyPair(cell.pair)} seed ${cell.seed} lambda ${cell.lam}`}
    >
      <img src={thumbUrl(cell.image.relPath)} alt={label ?? cell.image.relPath} loading="lazy" />
      <span className={`verdictbadge ${cell.compose ? 'compose' : ''}`}>
        {cell.nInstances} {cell.nInstances === 1 ? 'animal' : 'animals'}
        {cell.compose ? ' · compose' : ''}
      </span>
      {cell.image.sharedAtZero && <span className="shared">shared row</span>}
    </button>
  )
}

/** A small cell for riding a table row. Border says the scorer's verdict. */
export function MiniCell({ cell, px = 46 }: { cell: Cell | undefined; px?: number }) {
  const set = useScene((s) => s.set)
  if (!cell) return <span style={{ color: 'var(--ink-3)' }}>no cell</span>
  return (
    <img
      src={thumbUrl(cell.image.relPath)}
      alt={`${cell.pair} seed ${cell.seed} lambda ${cell.lam}`}
      width={px}
      height={px}
      loading="lazy"
      onClick={() => set({ lightbox: cell.image.relPath })}
      title={`${prettyPair(cell.pair)} seed ${cell.seed}, λ=${cell.lam}, ${cell.nInstances} detected`}
      style={{
        borderRadius: 5,
        objectFit: 'cover',
        cursor: 'zoom-in',
        display: 'block',
        border: `${cell.compose ? 3 : 1}px solid ${cell.compose ? 'var(--oracle)' : 'var(--rule)'}`,
      }}
    />
  )
}

export function Lightbox() {
  const lightbox = useScene((s) => s.lightbox)
  const set = useScene((s) => s.set)
  if (!lightbox) return null
  return (
    <div className="lightbox" onClick={() => set({ lightbox: null })}>
      <img src={fullUrl(lightbox)} alt={lightbox} />
      <div className="cap">
        {D.meta.imagesRoot}/{lightbox}
      </div>
    </div>
  )
}
