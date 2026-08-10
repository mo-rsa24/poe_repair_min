import { childrenOf, fmtParams, fmtShape, M, paramShare } from "./data";
import { useScene } from "./store";

/** Container levels: down/up/mid blocks, samplers, ModuleLists, and the
 *  generic block card. Children as descendable cards, chain drawn in order. */
export function VBlock({ path }: { path: string }) {
  const navigate = useScene((s) => s.navigate);
  const setHover = useScene((s) => s.setHover);
  const m = M[path];
  const kids = childrenOf(path);
  const isMid = m.cls === "UNetMidBlock2D";
  const isDown = m.cls === "DownEncoderBlock2D";
  const isUp = m.cls === "UpDecoderBlock2D";
  const isSampler = m.cls === "Downsample2D" || m.cls === "Upsample2D";

  // execution order for the chain figure
  let chain: string[] = kids;
  if (isMid) {
    const r = kids.filter((k) => k.endsWith("resnets"));
    const a = kids.filter((k) => k.endsWith("attentions"));
    if (r.length && a.length) {
      chain = [r[0] + ".0", a[0] + ".0", r[0] + ".1"];
    }
  } else if (isDown || isUp) {
    const rs = childrenOf(path + ".resnets");
    const smp = kids.find((k) => k.includes("samplers"));
    chain = [...rs, ...(smp ? childrenOf(smp) : [])];
  } else if (m.cls === "ModuleList") {
    chain = kids;
  }
  const boxes = chain.filter((c) => M[c]);

  const note = isMid ? (
    <>The bottleneck sandwich: resnet, then the VAE's only attention (every one of the 128² = 16384
      latent positions attends to every other), then a second resnet. Spatial size never changes here;
      this block is where the latent gets its global coherence.</>
  ) : isDown ? (
    <>Two resnets refine at constant size, then the downsampler halves H and W with a stride-2 conv.
      Note the asymmetric pad: diffusers pads (0,1,0,1) by hand, so the conv sees {fmtShape(M[path + ".downsamplers.0.conv"]?.shapes?.in)} rather than a p=1 ring.</>
  ) : isUp ? (
    <>Three resnets (one more than the encoder's blocks), then nearest-neighbour ×2 upsample followed
      by a 3×3 conv to clean the blockiness. The last up block (up_blocks.3) has no upsampler; it is already at 1024².</>
  ) : isSampler ? (
    <>A thin wrapper: {m.cls === "Downsample2D" ? "manual (0,1,0,1) pad, then one stride-2 conv" : "F.interpolate nearest ×2, then one 3×3 conv"}. Descend into the conv for the stepper.</>
  ) : (
    <>Container of {kids.length} children, in registration order.</>
  );

  return (
    <div className="viz">
      <h2>{path} <span className="tag real">real</span></h2>
      <div className="stats">
        <span className="stat">{m.cls}</span>
        <span className="stat">params <b>{fmtParams(m.params)}</b> ({paramShare(m.params)})</span>
        {m.shapes && <span className="stat">in <b>{fmtShape(m.shapes.in)}</b> → out <b>{fmtShape(m.shapes.out)}</b></span>}
      </div>
      <p className="note">{note}</p>
      {boxes.length > 0 && (
        <div className="figure">
          <svg viewBox={`0 0 ${boxes.length * 170 + 20} 120`} width={boxes.length * 170 + 20} height={120} role="img">
            <defs>
              <marker id="ba" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                <path d="M0,0 L7,4 L0,8 Z" fill="var(--faint)" />
              </marker>
            </defs>
            {boxes.map((c, i) => {
              const cm = M[c];
              const short = c.slice(path.length + 1);
              const col = cm.cls === "Attention" ? "var(--cond)" : cm.cls.includes("Resnet") ? "var(--enc)" : "var(--dec)";
              const tok = short.split(".")[0];
              return (
                <g key={c} data-tok={tok} style={{ cursor: "pointer" }} onClick={() => navigate(c)}
                  onMouseEnter={() => setHover(tok)} onMouseLeave={() => setHover(null)}>
                  <rect x={10 + i * 170} y={30} width={150} height={56} rx={9}
                    fill={`color-mix(in srgb, ${col} 16%, var(--surface))`} stroke={col} strokeWidth={1.6} />
                  <text x={85 + i * 170} y={52} textAnchor="middle" fontSize={10.5} className="m" fill="var(--ink)" fontWeight={600}>{short}</text>
                  <text x={85 + i * 170} y={67} textAnchor="middle" fontSize={9} className="m" fill="var(--soft)">{cm.cls} · {fmtParams(cm.params)}</text>
                  <text x={85 + i * 170} y={80} textAnchor="middle" fontSize={8.5} className="m" fill="var(--faint)">out {fmtShape(cm.shapes?.out)}</text>
                  {i < boxes.length - 1 && (
                    <line x1={160 + i * 170} y1={58} x2={178 + i * 170} y2={58} stroke="var(--faint)" strokeWidth={2} markerEnd="url(#ba)" />
                  )}
                </g>
              );
            })}
            <text x={12} y={16} fontSize={10} fill="var(--soft)">execution order (traced) · click a box to descend</text>
          </svg>
        </div>
      )}
      <div className="kids">
        {kids.map((k) => (
          <button className="kid" key={k} onClick={() => navigate(k)}>
            <div className="k-name">{k.slice(path.length + 1)}</div>
            <div className="k-meta">{M[k].cls} · {fmtParams(M[k].params)}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
