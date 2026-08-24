import { useEffect } from 'react'
import { D, useScene } from './store'
import { ClaimView, claimSummary } from './ClaimViews'
import { Lightbox } from './parts'

const SHORT: Record<string, string> = {
  C1: 'the dose response',
  C2: 'equal cells',
  C3: 'the size floor',
  C4: 'the harness',
  C5: 'the controls differ',
  C6: 'eye against scorer',
  C7: 'the strip',
  C8: 'the missing control',
  C9: 'where it landed',
}

function Masthead() {
  const verdict = D.verdict.body.join(' ').replace(/\*\*/g, '')
  const quoted = D.claims.filter((c) => c.state === 'quoted').length
  return (
    <div className="masthead">
      <div className="kicker">
        register slot F2 · {(D.runKind.text.match(/\*\*(.+?)\*\*/)?.[1] ?? D.runKind.text).replace(/\.$/, '')} ·
        scorer {D.fileSummary.scorer}
      </div>
      <h1>Does more correction give more composition?</h1>
      <p className="verdict">{verdict}</p>
      <p className="verdict" style={{ marginTop: 10, fontSize: 13.5 }}>
        {D.cells.length} scored cells and {D.meta.thumbs?.unique ?? '?'} images are loaded from
        disk. Every rate on these pages is recomputed here and checked against the results file.{' '}
        {quoted} of {D.claims.length} claims have no file behind their numbers and are drawn apart
        from the rest.
      </p>
    </div>
  )
}

function Strip() {
  const current = useScene((s) => s.claim)
  const set = useScene((s) => s.set)
  return (
    <nav className="strip" aria-label="claims">
      {claimSummary().map((c) => (
        <button key={c.id} className="chip" aria-current={c.id === current}
          onClick={() => set({ claim: c.id })} title={c.title}>
          <span className={`dot ${c.state === 'quoted' ? 'quoted' : ''} ${c.flagged ? 'flagged' : ''}`} />
          {c.id} · {SHORT[c.id]}
        </button>
      ))}
    </nav>
  )
}

function Footer() {
  const b = D.meta.builtFrom
  return (
    <div className="panel" style={{ background: 'transparent', boxShadow: 'none' }}>
      <h3 style={{ marginTop: 0 }}>What this page read</h3>
      <div className="prov" style={{ borderTop: 0, paddingTop: 0 }}>
        {Object.entries(b).map(([k, v]) => (
          <div key={k}>
            <b>{k}</b> {v.path} <span style={{ opacity: 0.7 }}>({v.mtime})</span>
          </div>
        ))}
        <div style={{ marginTop: 8 }}>
          <b>run identity</b> {D.meta.runIdNote} Log: {D.meta.runLog}
        </div>
        <div>
          <b>images</b> {D.meta.imagesRoot} · thumbnails at {D.meta.thumbs?.px ?? '?'}px, full size
          on click
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const claim = useScene((s) => s.claim)
  const set = useScene((s) => s.set)
  const lam = useScene((s) => s.lam)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') set({ lightbox: null })
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        const i = D.lambdas.indexOf(lam)
        const next = e.key === 'ArrowRight' ? i + 1 : i - 1
        if (next >= 0 && next < D.lambdas.length) set({ lam: D.lambdas[next] })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lam, set])

  return (
    <div className="wrap">
      <Masthead />
      <Strip />
      <ClaimView id={claim} />
      <Footer />
      <Lightbox />
    </div>
  )
}
