import { Eq } from "./Eq";
import { childrenOf, fmtParams, fmtShape, M, paramShare } from "./data";
import { useScene } from "./store";

/** Attention: the one-head Q K V walk. Toy weights are illustrative and
 *  labelled; the head-split arithmetic and shapes are real. */
export function VAttention({ path }: { path: string }) {
  const navigate = useScene((s) => s.navigate);
  const setHover = useScene((s) => s.setHover);
  const step = useScene((s) => s.step);
  const setStep = useScene((s) => s.setStep);
  const m = M[path];
  const heads = (m.args.heads as number) ?? 1;
  const dimHead = (m.args.dim_head as number) ?? 512;
  const C = heads * dimHead;
  const H = (m.shapes?.in as number[])?.[2] ?? 128;
  const tokens = H * H;
  const kids = childrenOf(path);

  // toy example: 5 tokens, fixed logits per query, softmaxed live (real arithmetic, illustrative values)
  const N = 5;
  const qi = step % N;
  const logits = Array.from({ length: N }, (_, j) => Math.sin((qi + 1) * (j + 1) * 1.3) * 2);
  const exps = logits.map((l) => Math.exp(l));
  const Z = exps.reduce((a, b) => a + b, 0);
  const w = exps.map((e) => e / Z);
  const rowSum = w.reduce((a, b) => a + b, 0);

  const colQ = "var(--enc)", colK = "var(--cond)", colV = "var(--dec)";
  const y0 = 46, dy = 34;

  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real shapes</span> <span className="tag illus">illustrative weights</span></h2>
      <div className="stats">
        <span className="stat">Attention · <b>{fmtParams(m.params)}</b> ({paramShare(m.params)})</span>
        <span className="stat">heads <b>{heads}</b> · d_head <b>{dimHead}</b> · C = {heads}×{dimHead} = <b>{C}</b></span>
        <span className="stat">in <b>{fmtShape(m.shapes?.in)}</b> → {tokens.toLocaleString()} tokens of dim {C}</span>
      </div>
      <p className="note">
        The VAE's only attention. The 512×{H}×{H} feature map is flattened to <b>{tokens.toLocaleString()} tokens</b>,
        each projected to a query, key, and value by to_q/to_k/to_v; every latent position attends to every other
        (a {tokens.toLocaleString()}×{tokens.toLocaleString()} map, the model's one global operation).
        Below: the same mechanics on 5 toy tokens, one head, one query at a time. Weights are illustrative; the softmax arithmetic is computed live.
      </p>
      <div className="figure">
        <svg viewBox="0 0 660 240" role="img" style={{ width: "100%", height: "auto", minWidth: 640 }}>
          <text x={60} y={26} textAnchor="middle" fontSize={11} fill={colQ} fontWeight={700}>queries</text>
          <text x={250} y={26} textAnchor="middle" fontSize={11} fill={colK} fontWeight={700}>keys / values</text>
          {Array.from({ length: N }).map((_, i) => (
            <g key={"q" + i} style={{ cursor: "pointer" }} onClick={() => setStep(i)}
              onMouseEnter={() => setHover("query")} onMouseLeave={() => setHover(null)}>
              <rect x={26} y={y0 + i * dy} width={66} height={24} rx={6}
                fill={i === qi ? `color-mix(in srgb, ${colQ} 34%, var(--surface))` : "var(--surface)"}
                stroke={colQ} strokeWidth={i === qi ? 2.2 : 1} />
              <text x={59} y={y0 + i * dy + 16} textAnchor="middle" fontSize={10.5} className="m" fill="var(--ink)">q{i}</text>
            </g>
          ))}
          {Array.from({ length: N }).map((_, j) => (
            <g key={"k" + j} onMouseEnter={() => setHover("key")} onMouseLeave={() => setHover(null)}>
              <rect x={216} y={y0 + j * dy} width={66} height={24} rx={6} fill="var(--surface)" stroke={colK} />
              <text x={249} y={y0 + j * dy + 16} textAnchor="middle" fontSize={10.5} className="m" fill="var(--ink)">k{j}·v{j}</text>
            </g>
          ))}
          {w.map((wj, j) => (
            <line key={"e" + j} x1={92} y1={y0 + qi * dy + 12} x2={216} y2={y0 + j * dy + 12}
              stroke={colQ} strokeWidth={0.8 + wj * 7} opacity={0.25 + wj * 0.75} />
          ))}
          <g onMouseEnter={() => setHover("softmax")} onMouseLeave={() => setHover(null)}>
            <circle cx={340} cy={y0 + 2 * dy + 12} r={20} fill="var(--surface)" stroke={colV} strokeWidth={2} />
            <text x={340} y={y0 + 2 * dy + 17} textAnchor="middle" fontSize={12} fill={colV} fontWeight={700}>Σ</text>
          </g>
          {w.map((wj, j) => (
            <line key={"s" + j} x1={282} y1={y0 + j * dy + 12} x2={321} y2={y0 + 2 * dy + 12}
              stroke={colV} strokeWidth={0.8 + wj * 7} opacity={0.25 + wj * 0.75} />
          ))}
          <line x1={360} y1={y0 + 2 * dy + 12} x2={392} y2={y0 + 2 * dy + 12} stroke="var(--faint)" strokeWidth={2} />
          <rect x={394} y={y0 + 2 * dy - 2} width={64} height={28} rx={6} fill={`color-mix(in srgb, ${colV} 20%, var(--surface))`} stroke={colV} />
          <text x={426} y={y0 + 2 * dy + 16} textAnchor="middle" fontSize={10.5} className="m" fill="var(--ink)">out q{qi}</text>
          {/* the map as matrix, active row highlighted */}
          <text x={560} y={26} textAnchor="middle" fontSize={11} fill="var(--soft)" fontWeight={700}>the map, row q{qi}</text>
          {Array.from({ length: N }).map((_, r) =>
            Array.from({ length: N }).map((_, c) => {
              const active = r === qi;
              const val = active ? w[c] : 0.12;
              return (
                <rect key={"m" + r + "," + c} x={500 + c * 26} y={40 + r * 26} width={24} height={24} rx={3}
                  fill={`color-mix(in srgb, ${colQ} ${Math.round(val * 90)}%, var(--surface2))`}
                  stroke={active ? colQ : "var(--line)"} strokeWidth={active ? 1.6 : 1} opacity={active ? 1 : 0.45} />
              );
            })
          )}
          {w.map((wj, c) => (
            <text key={"t" + c} x={512 + c * 26} y={40 + qi * 26 + 16} textAnchor="middle" fontSize={7.5} className="m"
              fill="var(--ink)">{wj.toFixed(2)}</text>
          ))}
          <text x={560} y={40 + N * 26 + 16} textAnchor="middle" fontSize={9.5} className="m" fill={Math.abs(rowSum - 1) < 1e-9 ? "var(--dec)" : "var(--bad)"}>
            row sum = {rowSum.toFixed(4)} ✓
          </text>
        </svg>
      </div>
      <div className="ctl">
        <label>query</label>
        <input type="range" min={0} max={N - 1} value={qi} onChange={(e) => setStep(Number(e.target.value))} />
        <span className="readout">q{qi} · weights softmax(logits), live</span>
      </div>
      <Eq block tex={String.raw`\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V,\qquad d=${dimHead},\; C=${heads}\times${dimHead}=${C}`} />
      <p className="note">
        Hover the columns to light to_q / to_k / to_v handling in AttnProcessor2_0 (second code tab).
        In the real module this is one head of width 512, so the "split" is trivial: C = 1×512.
      </p>
      <div className="kids">
        {kids.map((k) => (
          <button className="kid" key={k} onClick={() => navigate(k)}>
            <div className="k-name">{k.slice(path.length + 1)}</div>
            <div className="k-meta">{M[k].cls} · {fmtParams(M[k].params)} · out {fmtShape(M[k].shapes?.out)}</div>
          </button>
        ))}
      </div>
      <p className="drip">
        want attention derivable from first principles? <code>/drip --math attention</code>, then <code>/polish</code>, then <code>/math-scene</code>.
      </p>
    </div>
  );
}
