import type { ComponentType } from 'react'
import { Lightbox, plainText, Quoted } from './parts'
import { C1, C2, C3, C4, C5, C6, C7, Owed } from './ClaimViews'
import { useShallow } from 'zustand/react/shallow'
import { D, useScene } from './store'

const VIEWS: Record<string, ComponentType> = {
  C1, C2, C3, C4, C5, C6, C7,
}

export default function App() {
  const { claim, set } = useScene(useShallow((s) => ({ claim: s.claim, set: s.set })))
  const current = D.claims.find((c) => c.id === claim)
  const View = VIEWS[claim]

  return (
    <div className="app">
      <aside className="rail">
        <h1>What the cached runs already show</h1>
        <div className="sub">
          four analyses over the cache, no sampling
          <br />built from commit {D.meta.commit}
        </div>

        {D.claims.map((c) => (
          <button key={c.id}
                  className={`claim-btn ${claim === c.id ? 'on' : ''} ${c.state === 'not-run' ? 'gap' : ''}`}
                  onClick={() => set({ claim: c.id })}>
            <span className="id">{c.id}</span>
            {c.mark ? `${c.mark} ` : ''}{c.question}
          </button>
        ))}

        <button className={`claim-btn ${claim === 'owed' ? 'on' : ''}`}
                onClick={() => set({ claim: 'owed' })}>
          <span className="id">—</span>owed, and where this page disagrees
        </button>
      </aside>

      <main className="main">
        <div className="verdict">
          {plainText(D.verdict.body.join(' '))}
          <div className="src">
            {D.verdict.path.split('/').slice(-2).join('/')} : {D.verdict.lines[0]}-{D.verdict.lines[1]}
            {' · '}run kind: {plainText(D.runKind.body.join(' '))}
          </div>
        </div>

        {claim === 'owed' ? (
          <>
            <h2>What is owed</h2>
            <Owed />
          </>
        ) : current && View ? (
          <>
            <h2>{current.mark ? `${current.mark} ` : ''}{current.question}</h2>
            <Quoted q={current.quote} />
            <View />
          </>
        ) : (
          <div className="slot">no such claim</div>
        )}

        <div className="panel" style={{ marginTop: 34 }}>
          <h3>Words this page uses, from the review file</h3>
          {D.vocabulary.map((v, i) => <Quoted key={i} q={v} />)}
        </div>
      </main>

      <Lightbox />
    </div>
  )
}
