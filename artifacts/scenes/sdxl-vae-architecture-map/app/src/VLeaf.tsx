import { fmtParams, fmtShape, M, paramShare } from "./data";
import { useScene } from "./store";

function Stats({ path }: { path: string }) {
  const m = M[path];
  return (
    <div className="stats">
      <span className="stat">{m.cls} · <b>{fmtParams(m.params)}</b>{m.params > 0 && <> ({paramShare(m.params)})</>}</span>
      {m.shapes && <span className="stat">in <b>{fmtShape(m.shapes.in)}</b> → out <b>{fmtShape(m.shapes.out)}</b> (shape-preserving)</span>}
    </div>
  );
}

/** GroupNorm: the group strip. 32 groups × ch/32, mean/var per group. */
export function VGroupNorm({ path }: { path: string }) {
  const setHover = useScene((s) => s.setHover);
  const m = M[path];
  const a = m.args as { groups: number; ch: number };
  const per = a.ch / a.groups;
  const show = Math.min(a.groups, 16);
  const cw = 34;
  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <Stats path={path} />
      <p className="note">
        GroupNorm splits the <b>{a.ch}</b> channels into <b>{a.groups}</b> groups of <b>{per}</b>, and normalises each
        group by its own mean and variance, computed over (channels-in-group × H × W). No batch statistics,
        which is why it behaves identically at batch 1 (your case, always). The learned γ, β are per-channel: 2×{a.ch} = {fmtParams(m.params)} params.
      </p>
      <div className="figure" onMouseEnter={() => setHover("F.group_norm")} onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${show * cw + 120} 120`} width={show * cw + 120} height={120} role="img">
          {Array.from({ length: show }).map((_, g) => (
            <g key={g}>
              <rect x={10 + g * cw} y={26} width={cw - 4} height={44} rx={5}
                fill={`color-mix(in srgb, var(--enc) ${12 + (g % 4) * 9}%, var(--surface))`} stroke="var(--line2)" />
              <text x={10 + g * cw + (cw - 4) / 2} y={52} textAnchor="middle" fontSize={8.5} className="m" fill="var(--soft)">{per}ch</text>
              <text x={10 + g * cw + (cw - 4) / 2} y={84} textAnchor="middle" fontSize={7.5} className="m" fill="var(--faint)">μ{g},σ{g}</text>
            </g>
          ))}
          {show < a.groups && <text x={14 + show * cw} y={52} fontSize={10} fill="var(--faint)">… ×{a.groups} groups total</text>}
          <text x={10} y={14} fontSize={10} fill="var(--soft)">{a.groups} groups × {per} channels each · every group normalised by its own μ, σ</text>
          <text x={10} y={108} fontSize={9.5} className="m" fill="var(--soft)">y = γ · (x − μ_g)/√(σ²_g + ε) + β · per-channel γ,β</text>
        </svg>
      </div>
    </div>
  );
}

/** SiLU chip: the gate curve, drawn from the real function. */
export function VSiLU({ path }: { path: string }) {
  const setHover = useScene((s) => s.setHover);
  const pts = Array.from({ length: 81 }, (_, i) => {
    const x = -5 + i * 0.125;
    const y = x / (1 + Math.exp(-x));
    return [40 + (x + 5) * 22, 90 - y * 16] as const;
  });
  const d = "M" + pts.map((p) => p.join(",")).join(" L");
  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <Stats path={path} />
      <p className="note">
        SiLU (swish): x · sigmoid(x). Smooth near zero, linear for large x, slightly negative dip at x ≈ −1.28.
        Zero parameters; it appears between every norm and conv in this model (act_fn "silu" in the config).
      </p>
      <div className="figure" onMouseEnter={() => setHover("silu")} onMouseLeave={() => setHover(null)}>
        <svg viewBox="0 0 300 120" width={300} height={120} role="img">
          <line x1={40} y1={90} x2={280} y2={90} stroke="var(--line2)" />
          <line x1={150} y1={10} x2={150} y2={110} stroke="var(--line2)" />
          <path d={d} fill="none" stroke="var(--cond)" strokeWidth={2.5} />
          <text x={286} y={94} fontSize={9} fill="var(--faint)">x</text>
          <text x={150} y={116} textAnchor="middle" fontSize={9.5} className="m" fill="var(--soft)">silu(x) = x · σ(x)</text>
        </svg>
      </div>
    </div>
  );
}

/** Dropout: inert pass-through at inference, named rather than hidden. */
export function VDropout({ path }: { path: string }) {
  const m = M[path];
  const p = (m.args.p as number) ?? 0;
  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <Stats path={path} />
      <p className="note">
        Dropout with p = {p}. {p === 0 ? <>At p=0 this is the identity, always: diffusers instantiates it in every ResnetBlock2D but the VAE config never turns it on. It exists in the tree, so it is named here rather than hidden; it does nothing at inference or training.</> : <>Active only in training mode.</>}
      </p>
      <div className="figure">
        <svg viewBox="0 0 300 70" width={300} height={70} role="img">
          <rect x={20} y={20} width={80} height={30} rx={7} fill="var(--surface)" stroke="var(--line2)" />
          <text x={60} y={39} textAnchor="middle" fontSize={10.5} className="m" fill="var(--soft)">x</text>
          <line x1={100} y1={35} x2={180} y2={35} stroke="var(--faint)" strokeWidth={2} strokeDasharray="6 4" />
          <text x={140} y={26} textAnchor="middle" fontSize={9} fill="var(--faint)">pass-through (p={p})</text>
          <rect x={180} y={20} width={80} height={30} rx={7} fill="var(--surface)" stroke="var(--line2)" />
          <text x={220} y={39} textAnchor="middle" fontSize={10.5} className="m" fill="var(--soft)">x</text>
        </svg>
      </div>
    </div>
  );
}

/** Linear: the bipartite node-edge fan (attention's to_q/to_k/to_v/to_out).
 *  A handful of the real units drawn as nodes; click a left node to move
 *  the highlighted fan (the shared stepper clock). */
export function VLinear({ path }: { path: string }) {
  const setHover = useScene((s) => s.setHover);
  const step = useScene((s) => s.step);
  const setStep = useScene((s) => s.setStep);
  const m = M[path];
  const a = m.args as { din: number; dout: number };
  const role = path.endsWith("to_q") ? "queries" : path.endsWith("to_k") ? "keys" : path.endsWith("to_v") ? "values" : "output projection";
  const tok = path.split(".").pop() ?? "to_q";
  const N = 5; // visible nodes per column, plus the ellipsis row
  const sel = step % N;
  const y0 = 44, dy = 34, xL = 90, xR = 330, r = 11;
  const yAt = (i: number) => y0 + i * dy + (i >= 3 ? 26 : 0); // gap where the ⋮ sits
  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <Stats path={path} />
      <p className="note">
        The {role} projection: every one of the 16384 latent tokens is mapped {a.din}→{a.dout} by the same
        weight matrix; each output unit (right) reads <b>all {a.din}</b> inputs (left). Five of the {a.din} units
        are drawn; the fan shows one output unit's incoming weights. Click a right node to move it.
        {" "}params = {a.din}·{a.dout} + {a.dout} = {(a.din * a.dout + a.dout).toLocaleString()} {a.din * a.dout + a.dout === m.params ? "✓" : "✗"}.
      </p>
      <div className="figure" onMouseEnter={() => setHover(tok)} onMouseLeave={() => setHover(null)}>
        <svg viewBox="0 0 460 260" role="img" style={{ width: "100%", height: "auto", minWidth: 440 }}>
          <text x={xL} y={22} textAnchor="middle" fontSize={11} fill="var(--enc)" fontWeight={700}>in · {a.din}</text>
          <text x={xR} y={22} textAnchor="middle" fontSize={11} fill="var(--dec)" fontWeight={700}>out · {a.dout}</text>
          {/* edges: all-to-all at whisper opacity, the selected unit's fan in accent */}
          {Array.from({ length: N }).map((_, i) =>
            Array.from({ length: N }).map((_, j) => {
              const hot = j === sel;
              return (
                <line key={i + "-" + j} x1={xL + r} y1={yAt(i)} x2={xR - r} y2={yAt(j)}
                  stroke={hot ? "var(--accent)" : "var(--line2)"} strokeWidth={hot ? 1.6 : 0.8}
                  opacity={hot ? 0.85 : 0.5} />
              );
            })
          )}
          {Array.from({ length: N }).map((_, i) => (
            <circle key={"l" + i} cx={xL} cy={yAt(i)} r={r} fill={`color-mix(in srgb, var(--enc) 18%, var(--surface))`} stroke="var(--enc)" strokeWidth={1.5} />
          ))}
          {Array.from({ length: N }).map((_, j) => (
            <g key={"r" + j} style={{ cursor: "pointer" }} onClick={() => setStep(j)}>
              <circle cx={xR} cy={yAt(j)} r={r} fill={j === sel ? "var(--accent)" : `color-mix(in srgb, var(--dec) 18%, var(--surface))`}
                stroke={j === sel ? "var(--accent)" : "var(--dec)"} strokeWidth={1.5} />
            </g>
          ))}
          <text x={xL} y={y0 + 3 * dy + 8} textAnchor="middle" fontSize={14} fill="var(--faint)">⋮</text>
          <text x={xR} y={y0 + 3 * dy + 8} textAnchor="middle" fontSize={14} fill="var(--faint)">⋮</text>
          <text x={(xL + xR) / 2} y={yAt(N - 1) + 34} textAnchor="middle" fontSize={10} className="m" fill="var(--soft)">
            unit {sel}: out[{sel}] = Σᵢ W[{sel},i] · in[i] + b[{sel}] · {a.din} terms
          </text>
        </svg>
      </div>
    </div>
  );
}
