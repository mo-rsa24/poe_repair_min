import { childrenOf, fmtParams, fmtShape, M, paramShare } from "./data";
import { useScene } from "./store";

/** ResnetBlock2D: the residual chain with the skip arc, 1×1 shortcut when
 *  channels change, channel counts at the join. */
export function VResBlock({ path }: { path: string }) {
  const navigate = useScene((s) => s.navigate);
  const setHover = useScene((s) => s.setHover);
  const m = M[path];
  const kids = childrenOf(path);
  const cin = (m.shapes?.in as number[])?.[1] ?? 0;
  const cout = (m.shapes?.out as number[])?.[1] ?? 0;
  const hasShortcut = kids.some((k) => k.endsWith("conv_shortcut"));
  const seq = ["norm1", "nonlinearity", "conv1", "norm2", "nonlinearity", "conv2"];
  const labels: Record<string, string> = { norm1: "GroupNorm", norm2: "GroupNorm", nonlinearity: "SiLU", conv1: "Conv 3×3", conv2: "Conv 3×3" };
  const colors: Record<string, string> = { norm1: "var(--soft)", norm2: "var(--soft)", nonlinearity: "var(--cond)", conv1: "var(--enc)", conv2: "var(--enc)" };
  const Y = 88;
  const bw = 84, gap = 14;
  const W = 20 + seq.length * (bw + gap) + 60;

  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <div className="stats">
        <span className="stat">ResnetBlock2D · <b>{fmtParams(m.params)}</b> ({paramShare(m.params)})</span>
        <span className="stat">in <b>{fmtShape(m.shapes?.in)}</b> → out <b>{fmtShape(m.shapes?.out)}</b></span>
        <span className="stat">{hasShortcut ? "1×1 conv_shortcut (channels change)" : "identity skip (channels equal)"}</span>
      </div>
      <p className="note">
        Norm, SiLU, conv, twice, and the input rides over the top to be <b>added</b> at the join.
        {hasShortcut
          ? <> Channels change {cin}→{cout}, so the skip cannot be identity: a 1×1 conv projects it. That projection is the only place the skip touches weights.</>
          : <> Channels stay {cin}→{cout}, so the skip is a plain add: zero parameters, gradient flows through untouched.</>}
      </p>
      <div className="figure">
        <svg viewBox={`0 0 ${W} 170`} role="img" style={{ width: "100%", height: "auto", minWidth: W }}>
          <defs>
            <marker id="rba" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L7,4 L0,8 Z" fill="var(--faint)" />
            </marker>
          </defs>
          {seq.map((s, i) => {
            const x = 14 + i * (bw + gap);
            return (
              <g key={s + i} data-tok={s} style={{ cursor: "pointer" }}
                onMouseEnter={() => setHover("self." + s)} onMouseLeave={() => setHover(null)}
                onClick={() => M[path + "." + s] && navigate(path + "." + s)}>
                <rect x={x} y={Y} width={bw} height={36} rx={7} fill="var(--surface)" stroke={colors[s]} strokeWidth={1.5} />
                <text x={x + bw / 2} y={Y + 16} textAnchor="middle" fontSize={10.5} fill={colors[s]} fontWeight={600}>{labels[s]}</text>
                <text x={x + bw / 2} y={Y + 29} textAnchor="middle" fontSize={8.5} className="m" fill="var(--faint)">{s}</text>
                {i < seq.length - 1 && (
                  <line x1={x + bw} y1={Y + 18} x2={x + bw + gap - 2} y2={Y + 18} stroke="var(--faint)" strokeWidth={1.8} markerEnd="url(#rba)" />
                )}
              </g>
            );
          })}
          {(() => {
            const xEnd = 14 + seq.length * (bw + gap) + 8;
            const mid = 14 + (seq.length * (bw + gap)) / 2;
            return (
              <g>
                <circle cx={xEnd + 12} cy={Y + 18} r={13} fill="var(--surface)" stroke="var(--dec)" strokeWidth={2} />
                <text x={xEnd + 12} y={Y + 23} textAnchor="middle" fontSize={15} fill="var(--dec)" fontWeight={700}>+</text>
                {hasShortcut ? (
                  <g data-tok="conv_shortcut" style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover("conv_shortcut")} onMouseLeave={() => setHover(null)}
                    onClick={() => navigate(path + ".conv_shortcut")}>
                    <path d={`M 16 ${Y - 6} Q ${mid * 0.6} ${Y - 62} ${mid - 48} ${Y - 48}`} fill="none" stroke="var(--dec)" strokeWidth={2} strokeDasharray="5 4" />
                    <rect x={mid - 48} y={Y - 62} width={96} height={26} rx={6} fill={`color-mix(in srgb, var(--dec) 14%, var(--surface))`} stroke="var(--dec)" />
                    <text x={mid} y={Y - 45} textAnchor="middle" fontSize={10.5} className="m" fill="var(--dec)" fontWeight={600}>conv_shortcut 1×1</text>
                    <path d={`M ${mid + 48} ${Y - 48} Q ${xEnd * 0.9} ${Y - 50} ${xEnd + 12} ${Y + 3}`} fill="none" stroke="var(--dec)" strokeWidth={2} strokeDasharray="5 4" markerEnd="url(#rba)" />
                  </g>
                ) : (
                  <path d={`M 16 ${Y - 6} Q ${mid} ${Y - 66} ${xEnd + 12} ${Y + 3}`} fill="none" stroke="var(--dec)" strokeWidth={2} strokeDasharray="5 4" markerEnd="url(#rba)" />
                )}
                <text x={mid} y={Y - (hasShortcut ? 72 : 74)} textAnchor="middle" fontSize={10.5} fill="var(--dec)" fontWeight={600}>
                  {hasShortcut ? `skip: ${cin}→${cout} via 1×1` : "identity skip"}
                </text>
                <text x={xEnd + 12} y={Y + 48} textAnchor="middle" fontSize={9.5} className="m" fill="var(--soft)">{cout} ch at the join</text>
              </g>
            );
          })()}
        </svg>
      </div>
      <p className="note">Hover a box to light its line in ResnetBlock2D.forward on the right; click conv1/conv2 for the kernel stepper, a norm for the group strip.</p>
      <div className="kids">
        {kids.map((k) => (
          <button className="kid" key={k} onClick={() => navigate(k)}>
            <div className="k-name">{k.slice(path.length + 1)}</div>
            <div className="k-meta">{M[k].cls} · {fmtParams(M[k].params)} · out {fmtShape(M[k].shapes?.out)}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
